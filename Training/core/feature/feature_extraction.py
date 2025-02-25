import os
import multiprocessing
import collections
import warnings
import tempfile
import numpy as np
import tensorflow as tf
from ..tools import progressbar_wrapper, utils
from ..dataset_lib import build_dataset_from_numerical_data, build_dataset_via_generator
from ..config import logging, ErrorHandler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# from ...config import logging, ErrorHandler

logger = logging.getLogger('core.feature.feature_extraction')
logger.addHandler(ErrorHandler)


def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class FeatureExtraction(object):
    """Produce features for ML algorithms"""

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 file_ext=None,
                 update=False,
                 proc_number=2):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving meta information
        :param file_ext: file extent
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        self.naive_data_save_dir = naive_data_save_dir
        utils.mkdir(self.naive_data_save_dir)
        self.meta_data_save_dir = intermediate_save_dir
        utils.mkdir(self.meta_data_save_dir)
        self.file_ext = file_ext
        self.update = update
        self.proc_number = int(proc_number)

    def feature_extraction(self, sample_dir, use_order_features=False):
        """
        extract the android features from Android packages and save the extractions into designed directory
        :param sample_dir: malicious / benign samples for the subsequent process of feature extraction
        :param use_order_features: following the order of the provided sample paths
        """
        raise NotImplementedError

    def feature_preprocess(self, feature_path_list, gt_labels, data_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        :param feature_path_list: feature paths produced by the method of feature_extraction
        :param gt_labels: corresponding ground truth labels
        """
        raise NotImplementedError

    def feature2ipt(self, feature_path_list, labels=None, is_training_set=False, data_type='base'):
        """
        Mapping features to the input space

        :param feature_path_list, a list of paths point to the features
        :param labels, ground truth labels
        :param is_training_set, boolean type
        """
        raise NotImplementedError

    @staticmethod
    def _check(sample_dir):
        """
        check a valid directory and produce a list of file paths
        """
        if isinstance(sample_dir, str):
            if not os.path.exists(sample_dir):
                print(sample_dir)
                MSG = "No such directory or file {} exists!".format(sample_dir)
                raise ValueError(MSG)
            elif os.path.isfile(sample_dir):
                sample_path_list = [sample_dir]
            elif os.path.isdir(sample_dir):
                sample_path_list = list(utils.retrive_files_set(sample_dir, "", ".apk|"))
                assert len(sample_path_list) > 0, 'No files'
            else:
                raise ValueError(" No such path {}".format(sample_dir))
        elif isinstance(sample_dir, list):
            sample_path_list = [path for path in sample_dir if os.path.isfile(path)]
        else:
            MSG = "A directory or a list of paths are allowed!"
            raise ValueError(MSG)

        return sample_path_list


class DrebinFeature(FeatureExtraction):
    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 file_ext='.drebin',
                 update=False,
                 proc_number=2):
        super(DrebinFeature, self).__init__(naive_data_save_dir,
                                            intermediate_save_dir,
                                            file_ext,
                                            update,
                                            proc_number)

    def feature_extraction(self, sample_dir, use_order_features=False):
        """
        drebin1 features
        :return: 2D list, [[a list of features from an apk],...,[a list of features from an apk]]
        """
        from ..feature.drebin.drebin import AxplorerMapping, get_drebin_feature

        sample_path_list = self._check(sample_dir)
        pool = multiprocessing.Pool(self.proc_number)
        pbar = progressbar_wrapper.ProgressBar()
        process_results = []
        tasks = []
        pmap = AxplorerMapping()

        for i, apk_path in enumerate(sample_path_list):
            sha256 = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256 + self.file_ext)
            if os.path.exists(save_path) and (not self.update):
                continue
            tasks.append(apk_path)
            process_results = pool.apply_async(get_drebin_feature,
                                               args=(apk_path, pmap, save_path),
                                               callback=pbar.CallbackForProgressBar)

        pool.close()
        if process_results:
            pbar.DisplayProgressBar(process_results, len(tasks), type='hour')
        pool.join()

        feature_path_list = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_path_list.append(save_path)
            else:
                warnings.warn("Fail to perform feature extraction for '{}'".format(apk_path))

        return feature_path_list

    def load_features(self, feature_path_list):
        """
        load features
        :param feature_path_list: feature paths produced by the method of feature_extraction
        :return: a list of features
        """
        from .drebin.drebin import wrapper_load_features
        feature_list = []
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(n_proc)
        for res in pool.imap(wrapper_load_features, feature_path_list):
            if not isinstance(res, Exception):
                feature_list.append(res)
            else:
                print(str(res))
        return feature_list

    def feature_preprocess(self, feature_path_list, gt_labels, data_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        :param feature_path_list: feature paths produced by the method of feature_extraction
        :param gt_labels: corresponding ground truth labels

        """
        vocab_path = os.path.join(self.meta_data_save_dir, 'drebin_' + data_type + '.vocab')
        if self.update or (not os.path.exists(vocab_path)):
            assert len(feature_path_list) == len(gt_labels)
            features = self.load_features(feature_path_list)
            tmp_vocab = self.get_vocabulary(features)
            # we select 10,000 features
            selected_vocab = self.feature_selection(features, gt_labels, tmp_vocab, dim=10000)
            utils.dump_pickle(selected_vocab, vocab_path)
            print('save vocab to ' + vocab_path)
        return

    def feature2ipt(self, feature_path_list, labels=None, is_training_set=False, data_type='base'):
        """
        Mapping features to the input space
        :param feature_path_list: the feature paths produced by the method of feature_extraction
        :param labels: the ground truth labels correspond to features
        :param is_training_set, not used here
        :return tf.data; input dimension of an item of data
        :rtype tf.data.Dataset object; integer
        """
        # load
        vocab_path = os.path.join(self.meta_data_save_dir, 'drebin_' + data_type + '.vocab')

        if not os.path.exists(vocab_path):
            if labels is not None:
                self.feature_preprocess(feature_path_list, labels, data_type)
            else:
                raise ValueError('Need ground truth label!')
        vocab = utils.read_pickle(vocab_path)
        print('read vocab from ' + vocab_path)
        dim = len(vocab)
        print(dim)
        features = self.load_features(feature_path_list)
        dataX_np = self.get_feature_representation(features, vocab)
        if labels is not None:
            return build_dataset_from_numerical_data((dataX_np, labels)), dim, dataX_np
        else:
            return build_dataset_from_numerical_data(dataX_np), dim, dataX_np

    def feature_selection(self, train_features, train_y, vocab, dim):
        """
        feature selection
        :param train_features: 2D feature
        :type train_features: numpy object
        :param train_y: ground truth labels
        :param vocab: a list of words (i.e., features)
        :param dim: the number of remained words
        :return: chose vocab
        """
        is_malware = (train_y == 1)
        mal_features = np.array(train_features, dtype=object)[is_malware]
        ben_features = np.array(train_features, dtype=object)[~is_malware]

        if (len(mal_features) <= 0) or (len(ben_features) <= 0):
            return vocab

        mal_representations = self.get_feature_representation(mal_features, vocab)
        mal_frequency = np.sum(mal_representations, axis=0) / float(len(mal_features))
        ben_representations = self.get_feature_representation(ben_features, vocab)
        ben_frequency = np.sum(ben_representations, axis=0) / float(len(ben_features))

        # eliminate the words showing zero occurrence in apk files
        is_null_feature = np.all(mal_representations == 0, axis=0) & np.all(ben_representations, axis=0)
        mal_representations, ben_representations = None, None
        vocab_filtered = list(np.array(vocab)[~is_null_feature])

        if len(vocab_filtered) <= dim:
            return vocab_filtered
        else:
            feature_frq_diff = np.abs(mal_frequency[~is_null_feature] - ben_frequency[~is_null_feature])
            position_flag = np.argsort(feature_frq_diff)[::-1][:dim]

            vocab_selected = []
            for p in position_flag:
                vocab_selected.append(vocab_filtered[p])
            return vocab_selected

    def load_vocabulary(self, data_type):
        vocab_path = os.path.join(self.meta_data_save_dir, 'drebin_' + data_type + '.vocab')
        if not os.path.exists(vocab_path):
            raise ValueError("A vocabulary is needed.")
        vocab = utils.read_pickle(vocab_path)
        return vocab

    @staticmethod
    def get_vocabulary(feature_list, n=300000):
        """
        obtain the vocabulary based on the feature
        :param feature_list: 2D list of naive feature
        :param n: the number of top frequency items
        :return: feature vocabulary
        """
        c = collections.Counter()

        for features in feature_list:
            for feature in features:
                c[feature] = c[feature] + 1

        vocab, count = zip(*c.most_common(n))
        return list(vocab)

    @staticmethod
    def get_feature_representation(feature_list, vocab):
        """
        mapping feature to numerical representation
        :param feature_list: 2D feature list with shape [number of files, number of feature]
        :param vocab: a list of words
        :return: 2D representation
        :rtype numpy.ndarray
        """
        N = len(feature_list)
        M = len(vocab)

        assert N > 0 and M > 0

        representations = np.zeros((N, M), dtype=np.float32)
        dictionary = dict(zip(vocab, range(len(vocab))))
        for i, features in enumerate(feature_list):
            if len(features) > 0:
                filled_positions = [idx for idx in list(map(dictionary.get, features)) if idx is not None]
                if len(filled_positions) != 0:
                    representations[i, filled_positions] = 1.
                else:
                    warnings.warn("Produce zero feature vector.")

        return representations


class OpcodeSeq(FeatureExtraction):
    """
    get opcode sequences
    """

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir=None,
                 file_ext='.opcode',
                 update=False,
                 proc_number=2):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param file_ext: file extent
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        super(OpcodeSeq, self).__init__(naive_data_save_dir,
                                        intermediate_save_dir,
                                        file_ext,
                                        update,
                                        proc_number)

    def feature_extraction(self, sample_dir, use_order_features=False):
        from ..feature.opcodeseq.opcodeseq import feature_extr_wrapper

        sample_path_list = self._check(sample_dir)
        pool = multiprocessing.Pool(self.proc_number)
        pbar = progressbar_wrapper.ProgressBar()
        process_results = []
        tasks = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)

            if os.path.exists(save_path) and not self.update:
                continue
            tasks.append(apk_path)
            process_results = pool.apply_async(feature_extr_wrapper,
                                               args=(apk_path, save_path),
                                               callback=pbar.CallbackForProgressBar)

        pool.close()
        if process_results:
            pbar.DisplayProgressBar(process_results, len(tasks), type='hour')
        pool.join()

        feature_paths = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_paths.append(save_path)
            else:
                warnings.warn("Fail to perform feature extraction for '{}'".format(apk_path))

        return feature_paths

    def feature_preprocess(self, feature_path_list, gt_labels, data_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        :param feature_path_list: a list of paths directing to saved features
        :param gt_labels: corresponding ground truth labels
        """
        return

    def feature2ipt(self, feature_path_list, labels=None, is_training_set=False, data_type=None):
        """
        Mapping features to the input space
        """
        from ..feature.opcodeseq.opcodeseq import read_opcode
        from ..model_hp import text_cnn_hparam

        def padding_opcodes(features_of_an_apk, padding_char=0):
            padding_seq = []
            padding_chars = [padding_char] * text_cnn_hparam.kernel_size
            for i, seq in enumerate(features_of_an_apk):
                padding_seq.extend(seq)
                padding_seq.extend(padding_chars)
                if len(padding_seq) > text_cnn_hparam.max_sequence_length:
                    break
            return np.array(padding_seq[:text_cnn_hparam.max_sequence_length])

        def generator():
            try:
                if labels is not None:
                    for path, label in zip(feature_path_list, labels):
                        data_padded = padding_opcodes(read_opcode(path))
                        yield data_padded[:text_cnn_hparam.max_sequence_length], label
                else:
                    for path in feature_path_list:
                        data_padded = padding_opcodes(read_opcode(path))
                        yield data_padded[:text_cnn_hparam.max_sequence_length]
            except Exception as e:
                print(path)

        with tempfile.NamedTemporaryFile() as f:
            return build_dataset_via_generator(generator, labels, f.name), None, None


class APISequence(FeatureExtraction):
    """Obtain api sequences based on the function call graph"""

    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 use_feature_selection=True,
                 ratio=0.25,
                 file_ext='.seq',
                 update=False,
                 proc_number=2
                 ):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving meta information
        :param use_feature_selection: use feature selection to filtering out entities with high frequencies
        :param ratio: resides the range of [0, 1] and denotes a portion of features will be neglected
        :param file_ext: file extent
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        super(APISequence, self).__init__(naive_data_save_dir,
                                          intermediate_save_dir,
                                          file_ext,
                                          update,
                                          proc_number)
        self.use_feature_selection = use_feature_selection
        self.ratio = ratio
        from ..model_hp import droidetec_hparam
        self.maximum_vocab_size = droidetec_hparam.vocab_size

    def feature_extraction(self, sample_dir, use_order_features=False):
        """ save the android features and return the saved paths """
        from ..feature.apiseq.apiseq import get_api_sequence

        sample_path_list = self._check(sample_dir)
        pool = multiprocessing.Pool(self.proc_number)
        pbar = progressbar_wrapper.ProgressBar()
        process_results = []
        tasks = []

        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)

            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path) and (not self.update):
                continue
            tasks.append(apk_path)
            process_results = pool.apply_async(get_api_sequence,
                                               args=(apk_path, save_path),
                                               callback=pbar.CallbackForProgressBar)

        pool.close()
        if process_results:
            pbar.DisplayProgressBar(process_results, len(tasks), type='hour')
        pool.join()

        feature_paths = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_paths.append(save_path)
            else:
                warnings.warn("Fail to perform feature extraction for '{}'".format(apk_path))

        return feature_paths

    def feature_preprocess(self, feature_path_list, gt_labels, data_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        """
        dict_saving_path = os.path.join(self.meta_data_save_dir, 'apiseq_' + data_type + '.dict')
        if not os.path.exists(dict_saving_path) or self.update:
            vocab = self.get_vocab(feature_path_list, gt_labels)
            dictionary = dict(zip(vocab, range(len(vocab))))
            # saving
            utils.dump_joblib(dictionary, dict_saving_path)

        return

    def feature2ipt(self, feature_path_list, labels=None, is_training_set=False, data_type=None):
        """
        Mapping features to the input space
        """
        dict_saving_path = os.path.join(self.meta_data_save_dir, 'apiseq_' + data_type + '.dict')
        if os.path.exists(dict_saving_path):
            dictionary = utils.read_joblib(dict_saving_path)
        else:
            self.feature_preprocess(feature_path_list, labels, data_type)
            dictionary = utils.read_joblib(dict_saving_path)
        from ..model_hp import droidetec_hparam

        def generator():
            if labels is not None:
                discrete_features = self.feature_mapping(feature_path_list, dictionary)
                assert len(discrete_features) == len(labels), 'inconsistent data vs. corresponding label'
                for data, label in zip(*(discrete_features, labels)):
                    yield data[:droidetec_hparam.max_sequence_length], label
            else:
                from ..feature.apiseq.apiseq import wrapper_mapping
                for feature_path in feature_path_list:
                    data = wrapper_mapping([feature_path, dictionary])
                    if isinstance(data, Exception):
                        raise ValueError("Cannot load the specified feature:{}".format(
                            feature_path
                        ))
                    yield data[:droidetec_hparam.max_sequence_length]

        with tempfile.NamedTemporaryFile() as f:
            return build_dataset_via_generator(generator, labels, f.name), None, None

    def get_vocab(self, feature_path_list, gt_labels):
        """
        create vocabulary based on a list of feature
        :param feature_path_list: a list of feature paths
        :param gt_labels: ground truth labels
        :return: vocabulary
        :rtype: list
        """
        if self.use_feature_selection:
            assert 0. < self.ratio <= 1., 'the ratio should be (0,1]'

        from ..feature.apiseq import wrapper_load_feature
        c_mal = collections.Counter()
        c_ben = collections.Counter()

        # has side-effect, consuming lots of local disk
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(n_proc)
        for res, label in zip(pool.imap(wrapper_load_feature, feature_path_list), gt_labels):
            if isinstance(res, list):
                if label:
                    c_mal.update(res)
                else:
                    c_ben.update(res)
            elif isinstance(res, Exception):
                print(str(res))
            else:
                raise ValueError
        pool.close()
        pool.join()

        if not self.use_feature_selection:
            c_mal.update(c_ben)
            # c_all = c_mal
        else:
            api_num_mal = len(c_mal)
            api_hf_mal = dict(c_mal.most_common(int(api_num_mal * self.ratio))).keys()
            api_num_ben = len(c_ben)
            api_hf_ben = dict(c_ben.most_common(int(api_num_ben * self.ratio))).keys()
            common_apis = [e for e in api_hf_mal if e in api_hf_ben]
            for api in common_apis:
                c_mal[api] = 0
                c_ben[api] = 0
            c_mal.update(c_ben)
            logger.info('Filtering out {} apis.'.format(len(common_apis)))
        c_all = dict(c_mal.most_common(self.maximum_vocab_size - 1))  # saving a slot for null features
        c_all['sos'] = 1

        vocab, count = zip(*c_all.items())
        return list(vocab)

    def feature_mapping(self, feature_path_list, dictionary):
        """
        mapping feature to numerical representation
        :param feature_path_list: a list of feature paths
        :param dictionary: vocabulary -> index
        :return: 2D representation
        :rtype numpy.ndarray
        """
        numerical_features = []
        from ..feature.apiseq.apiseq import wrapper_mapping
        from ..model_hp import droidetec_hparam
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(n_proc)
        pargs = [(path, dictionary) for path in feature_path_list]
        for res, path in zip(pool.imap(wrapper_mapping, pargs), feature_path_list):
            if not isinstance(res, Exception):
                numerical_features.append(res[:droidetec_hparam.max_sequence_length])
            else:
                warnings.warn(str(res) + ': ' + path)
        pool.close()
        pool.join()
        return numerical_features


class MultiModality(FeatureExtraction):
    def __init__(self,
                 naive_data_save_dir,
                 intermediate_save_dir,
                 use_feature_selection=True,
                 feature_dimension=10000,
                 cluster_centers=100,
                 similar_threshold=0.5,
                 file_ext='.multimod',
                 update=False,
                 proc_number=2
                 ):
        """
        initialization
        :param naive_data_save_dir: a directory for saving intermediates
        :param intermediate_save_dir: a directory for saving meta information
        :param use_feature_selection: select features with top frequencies
        :param feature_dimension: the number of selected features, default 10,000
        :param cluster_centers: the number of cluster centers, default 100
        :param file_ext: file extent
        :param update: boolean indicator for recomputing the naive features
        :param proc_number: process number
        """
        super(MultiModality, self).__init__(naive_data_save_dir,
                                            intermediate_save_dir,
                                            file_ext,
                                            update,
                                            proc_number
                                            )
        self.use_feature_selection = use_feature_selection
        self.feature_dimension = feature_dimension
        self.cluster_centers = cluster_centers
        self.similar_threshold = similar_threshold

    def feature_extraction(self, sample_dir, use_order_features=False):
        """
        extract the android features from Android packages and save the extractions into designed directory
        """
        from ..feature.multimodality.multimodality import API_LIST, get_multimod_feature

        sample_path_list = self._check(sample_dir)
        pool = multiprocessing.Pool(self.proc_number)
        pbar = progressbar_wrapper.ProgressBar()
        process_results = []
        tasks = []

        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)

            if os.path.exists(save_path) and (not self.update):
                continue
            tasks.append(apk_path)
            process_results = pool.apply_async(get_multimod_feature,
                                               args=(apk_path, API_LIST, save_path),
                                               callback=pbar.CallbackForProgressBar)

        pool.close()
        if process_results:
            pbar.DisplayProgressBar(process_results, len(tasks), type='hour')
        pool.join()

        feature_paths = []
        for i, apk_path in enumerate(sample_path_list):
            sha256_code = os.path.splitext(os.path.basename(apk_path))[0]  # utils.get_sha256(apk_path)
            save_path = os.path.join(self.naive_data_save_dir, sha256_code + self.file_ext)
            if os.path.exists(save_path):
                feature_paths.append(save_path)
            else:
                warnings.warn("Fail to perform feature extraction for '{}'".format(apk_path))

        return feature_paths

    def feature_preprocess(self, feature_path_list, gt_labels, data_type):
        """
        pre-processing the naive data to accommodate the input format of ML algorithms
        :param feature_path_list: a list of paths directing to save features. For each apk,
        features produced by the method of feature_extraction, 2D list [[feature type 1,...,feature type 5],...,]
        :param gt_labels: corresponding ground truth labels
        """
        assert len(feature_path_list) == len(gt_labels), 'inconsistent dataset'
        vocab_path = os.path.join(self.meta_data_save_dir, 'multimodality_' + data_type + '.vocab')
        if os.path.exists(vocab_path) and not self.update:
            vocab_list = self.load_meta_info(vocab_path)
        else:
            vocab_list = self.get_vocab(feature_path_list,
                                        gt_labels,
                                        self.use_feature_selection,
                                        self.feature_dimension)
            # saving
            self.save_meta_info(vocab_list, vocab_path)

        dataX_list = self.feature_mapping(feature_path_list, vocab_list)

        # further processing
        scaler_path = os.path.join(self.meta_data_save_dir, 'multimodality_' + data_type + '.scaler')
        scaled_dataX_list = self.data_scaling(dataX_list, scaler_path)

        # clustering for last three types of features
        cluster_center_path = os.path.join(self.meta_data_save_dir, 'multimodality_' + data_type + '.center')
        if not os.path.exists(cluster_center_path) or self.update:
            cluster_centers = []
            for i, dataX in enumerate(scaled_dataX_list[2:]):  # produce the last three similarity-based feature
                center_vec = self.k_means_clustering(dataX,
                                                     self.cluster_centers)
                cluster_centers.append(center_vec)
            # saving
            self.save_meta_info(cluster_centers, cluster_center_path)
        return

    def feature2ipt(self, feature_path_list, labels=None, is_training_set=False, data_type=None):
        """
        Mapping features to the input space
        """
        # assert self._check_features(features)

        vocab_path = os.path.join(self.meta_data_save_dir, 'multimodality_' + data_type + '.vocab')
        if not os.path.isfile(vocab_path):
            self.feature_preprocess(feature_path_list, labels, data_type)
        vocab_list = self.load_meta_info(vocab_path)
        print('load vocab file: ' + vocab_path)
        scaler_path = os.path.join(self.meta_data_save_dir, 'multimodality_' + data_type + '.scaler')
        scalers = self.load_meta_info(scaler_path)
        print('load scalers file: ' + scaler_path)
        cluster_center_path = os.path.join(self.meta_data_save_dir, 'multimodality_' + data_type + '.center')
        centers = self.load_meta_info(cluster_center_path)
        print('load cluster center:' + cluster_center_path)

        dataX_list = self.feature_mapping(feature_path_list, vocab_list)
        for i, dataX in enumerate(dataX_list):
            dataX_list[i] = scalers[i].transform(dataX)

        for i, center in enumerate(centers):
            dataX_list[2 + i] = self._get_similarity(dataX_list[2 + i], center, self.similar_threshold)

        input_dim = []
        for x in dataX_list:
            input_dim.append(x.shape[1])

        # build dataset
        if labels is not None:
            data_tf = build_dataset_from_numerical_data(tuple(dataX_list))
            y = build_dataset_from_numerical_data(labels)
            from tensorflow import data
            return data.Dataset.zip((data_tf, y)), input_dim, dataX_list
        else:
            return dataX_list, input_dim, dataX_list

    def save_meta_info(self, data, path):
        if self.update or (not os.path.exists(path)):
            utils.dump_joblib(data, path)
        return

    @staticmethod
    def load_meta_info(path):
        if os.path.exists(path):
            return utils.read_joblib(path)
        else:

            raise ValueError("No such data.")

    @staticmethod
    def get_vocab(feature_path_list, gt_labels=None, use_feature_selection=False, dim=10000):
        """
        build vocabulary for five kinds of feature, including permission/component/environment, string, method api,
        method opcodes, shared library, each of which are presented in the 'collections.defaultdict' format
        :param feature_path_list: a list of paths redirecting to save features
        :param gt_labels: ground truth labels (optional)
        :param use_feature_selection: conducting feature extraction or not (False means no, and True means yes)
        :param dim: the number of selected feature (optional)
        :return: list of vocabularies corresponding to five kinds of feature
        """
        from ..feature.multimodality.multimodality import wrapper_load_features
        feature_list = []
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(n_proc)
        for res in pool.imap(wrapper_load_features, feature_path_list):
            if not isinstance(res, Exception):
                feature_list.append(res)
            else:
                print(str(res))

        assert isinstance(feature_list, list) and len(feature_list) > 0, 'Type: {} and length: {}'.format(
            type(feature_list), len(feature_list))

        number_of_types = len(feature_list[0])
        number_of_samples = len(feature_list)
        vocabulary_list = []
        for t in range(number_of_types):
            c = collections.Counter()
            for j in range(number_of_samples):
                feature_dict = feature_list[j][t]
                for k, v in feature_dict.items():
                    c[k] += v
            if not use_feature_selection:
                if len(c) > 0:
                    vocab, count = zip(*c.items())
                else:
                    vocab = []
            else:
                if len(c) > 0:
                    vocab, count = zip(*c.most_common(dim))  # filter out words with low frequency
                else:
                    vocab = []

            vocabulary_list.append(list(vocab))

        return vocabulary_list

    @staticmethod
    def feature_mapping(feature_path_list, vocab_list):
        """
        mapping feature to numerical representation
        :param feature_path_list: a list of paths redirecting to saved features
        :param vocab_list: several lists of words
        :return: 2D representation
        :rtype numpy.ndarray
        """
        from ..feature.multimodality.multimodality import wrapper_load_features
        feature_list = []
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(n_proc)
        for res in pool.imap(wrapper_load_features, feature_path_list):
            if not isinstance(res, Exception):
                feature_list.append(res)
            else:
                print(str(res))
        pool.close()
        pool.join()

        assert len(feature_list[0]) == len(vocab_list)
        number_of_feature_types = len(vocab_list)

        representation_list = []
        for t in range(number_of_feature_types):
            # feature_dict = np.array(feature_list)[:, t]
            number_of_samples = len(feature_list)
            vocab = vocab_list[t]
            M = len(list(vocab))
            representation = np.zeros((number_of_samples, M), dtype=np.float32)
            dictionary = dict(zip(vocab, range(M)))

            for j in range(number_of_samples):
                feature_dict = feature_list[j][t]
                if len(feature_dict) > 0:
                    filled_positions = [idx for idx in list(map(dictionary.get, list(feature_dict.keys()))) if
                                        idx is not None]
                    filled_values = [feature_dict.get(key) for key in list(feature_dict.keys()) if
                                     dictionary.get(key) is not None]
                    if len(filled_positions) != 0:
                        representation[j, filled_positions] = filled_values[:]
                    else:
                        warnings.warn("Produce zero feature vector.")
            representation_list.append(representation)
        return representation_list

    def data_scaling(self, data_x_list, scalar_saving_path=None):
        """
        minmax scaling for numerical feature representations
        :param data_x_list: a list of un-normalized feature representation
        :param scalar_saving_path:
        :return: scaled feature representation
        :rtype : list of 2d numpy.ndarray
        """
        if os.path.exists(scalar_saving_path) and not self.update:
            scalers = self.load_meta_info(scalar_saving_path)
        else:
            scalers = []
            for i, dataX in enumerate(data_x_list):
                scaler = MinMaxScaler()
                scaler.fit(dataX)
                scalers.append(scaler)
            self.save_meta_info(scalers, scalar_saving_path)
        for i, dataX in enumerate(data_x_list):
            data_x_list[i] = scalers[i].transform(dataX)

        return data_x_list

    @staticmethod
    def k_means_clustering(data_x, number_of_cluster_centers=100):
        N = data_x.shape[0]
        n_clusters = number_of_cluster_centers if number_of_cluster_centers < N else N // 2
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=0).fit(data_x)
        return kmeans.cluster_centers_

    @staticmethod
    def _get_similarity(data_x, anchor, threshold=0.5):
        """
        get similarity matrix
        :return: similarity-based feature representation
        """
        # The following method of calculating the similarity matrix might be different to the proposal
        # in the paper (i.e., Algorithm 2), which is confusing to us.
        similar_mat = 1. / np.column_stack(
            [np.max(np.square(data_x - center) ** 0.5 + 1, axis=-1) for center in anchor])
        return np.greater(similar_mat, threshold).astype(np.float32)

    @staticmethod
    def _check_features(features):
        """
        check the completeness
        :param features: a list of features, each item presented in the 'collections.defaultdict' format
        :return: True or False
        """
        return (isinstance(features, list)) and (len(features) > 0) and (
            isinstance(features[0][0], dict))
