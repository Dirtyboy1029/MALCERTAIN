from .vanilla import Vanilla, model_builder
from ..model_hp import train_hparam, mc_dropout_hparam
from ..tools import utils
from ..config import logging, ErrorHandler
import time

import tensorflow as tf

logger = logging.getLogger('ensemble.mc_dropout')
logger.addHandler(ErrorHandler)


class MCDropout(Vanilla):
    def __init__(self,
                 architecture_type='dnn',
                 base_model=None,
                 n_members=1,
                 model_directory=None,
                 name='MC_DROPOUT'
                 ):
        super(MCDropout, self).__init__(architecture_type,
                                        base_model,
                                        n_members,
                                        model_directory,
                                        name)
        self.hparam = utils.merge_namedtuples(train_hparam, mc_dropout_hparam)
        self.ensemble_type = 'mc_dropout'

    def build_model(self, input_dim=None):
        """
        Build an ensemble model -- only the homogeneous structure is considered
        :param input_dim: integer or list, input dimension shall be set in some cases under eager mode
        """
        callable_graph = model_builder(self.architecture_type)

        @callable_graph(input_dim, use_mc_dropout=True)
        def _builder():
            return utils.produce_layer(self.ensemble_type, dropout_rate=self.hparam.dropout_rate)

        self.base_model = _builder()
        return

    def model_generator(self):
        try:
            if len(self.weights_list) <= 0:
                self.load_ensemble_weights()
        except Exception as e:
            raise Exception("Cannot load model weights:{}.".format(str(e)))

        assert len(self.weights_list) == self.n_members
        self.base_model.set_weights(weights=self.weights_list[self.n_members - 1])
        # if len(self._optimizers_dict) > 0 and self.base_model.optimizer is not None:
        #     self.base_model.optimizer.set_weights(self._optimizers_dict[self.n_members - 1])
        for _ in range(self.hparam.n_sampling):
            yield self.base_model

    def fit(self, train_set, validation_set=None, input_dim=None, EPOCH=30, test_data=None, training_predict=True,
            **kwargs):
        """
        fit the ensemble by producing a lists of model weights
        :param train_set: tf.data.Dataset, the type shall accommodate to the input format of Tensorflow models
        :param validation_set: validation data, optional
        :param input_dim: integer or list, input dimension except for the batch size
        """
        # training preparation

        if self.base_model is None:
            self.build_model(input_dim=input_dim)

        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("The number of trainable variables: {}".format(len(self.base_model.trainable_variables)))
        logger.info("...training start!")

        best_val_accuracy = 0.
        total_time = 0.

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []

        for epoch in range(EPOCH):
            train_acc = 0.
            val_acc = 0.
            train_loss = 0.
            val_loss = 0.
            for member_idx in range(self.n_members):
                if member_idx < len(self.weights_list):  # loading former weights
                    self.base_model.set_weights(self.weights_list[member_idx])
                    self.base_model.optimizer.set_weights(self._optimizers_dict[member_idx])
                elif member_idx == 0:
                    pass  # do nothing
                else:
                    self.reinitialize_base_model()
                msg = 'Epoch {}/{}, member {}/{}, and {} member(s) in list'.format(epoch + 1,
                                                                                   EPOCH, member_idx + 1,
                                                                                   self.n_members,
                                                                                   len(self.weights_list))
                print(msg)
                start_time = time.time()
                history = self.base_model.fit(train_set,
                                              epochs=epoch + 1,
                                              initial_epoch=epoch,
                                              validation_data=validation_set
                                              )
                train_acc += history.history['binary_accuracy'][0]
                val_acc += history.history['binary_accuracy'][0]
                train_loss += history.history['loss'][0]
                val_loss += history.history['loss'][0]
                self.update_weights(member_idx,
                                    self.base_model.get_weights(),
                                    self.base_model.optimizer.get_weights())
                end_time = time.time()
                total_time += end_time - start_time

            # saving
            logger.info('Training ensemble costs {} seconds in total (including validation).'.format(total_time))
            train_acc = train_acc / self.n_members
            val_acc = val_acc / self.n_members
            train_loss = train_loss / self.n_members
            val_loss = val_loss / self.n_members
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            msg = 'Epoch {}/{}: training accuracy {:.5f}, validation accuracy {:.5f}.'.format(
                epoch + 1, EPOCH, train_acc, val_acc
            )
            logger.info(msg)

        self.save_ensemble_weights()

        return

    def finetune(self, train_set, validation_set=None, input_dim=None, EPOCH=30, test_data=None, training_predict=True,
            **kwargs):
        """
        fit the ensemble by producing a lists of model weights
        :param train_set: tf.data.Dataset, the type shall accommodate to the input format of Tensorflow models
        :param validation_set: validation data, optional
        :param input_dim: integer or list, input dimension except for the batch size
        """
        # training preparation

        self.base_model = None
        self.weights_list = []
        self._optimizers_dict = []
        self.load_ensemble_weights()

        self.base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate,
                                               clipvalue=self.hparam.clipvalue),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("The number of trainable variables: {}".format(len(self.base_model.trainable_variables)))
        logger.info("...training start!")

        best_val_accuracy = 0.
        total_time = 0.

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []

        for epoch in range(EPOCH):
            train_acc = 0.
            val_acc = 0.
            train_loss = 0.
            val_loss = 0.
            for member_idx in range(self.n_members):
                if member_idx < len(self.weights_list):  # loading former weights
                    self.base_model.set_weights(self.weights_list[member_idx])

                elif member_idx == 0:
                    pass  # do nothing
                else:
                    self.reinitialize_base_model()
                msg = 'Epoch {}/{}, member {}/{}, and {} member(s) in list'.format(epoch + 1,
                                                                                   EPOCH, member_idx + 1,
                                                                                   self.n_members,
                                                                                   len(self.weights_list))
                print(msg)
                start_time = time.time()
                history = self.base_model.fit(train_set,
                                              epochs=epoch + 1,
                                              initial_epoch=epoch,
                                              validation_data=validation_set
                                              )
                train_acc += history.history['binary_accuracy'][0]
                val_acc += history.history['binary_accuracy'][0]
                train_loss += history.history['loss'][0]
                val_loss += history.history['loss'][0]
                self.update_weights(member_idx,
                                    self.base_model.get_weights(),
                                    self.base_model.optimizer.get_weights())
                end_time = time.time()
                total_time += end_time - start_time

            # saving
            logger.info('Training ensemble costs {} seconds in total (including validation).'.format(total_time))
            train_acc = train_acc / self.n_members
            val_acc = val_acc / self.n_members
            train_loss = train_loss / self.n_members
            val_loss = val_loss / self.n_members
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            msg = 'Epoch {}/{}: training accuracy {:.5f}, validation accuracy {:.5f}.'.format(
                epoch + 1, EPOCH, train_acc, val_acc
            )
            logger.info(msg)

        return
