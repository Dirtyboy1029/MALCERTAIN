# MALCERTAIN


This code repository our ICSE 0204 paper titled **MALCERTAIN:Enhancing Deep Neural Network Based Android Malware Detection by Tackling Prediction Uncertainty**.
 
## Overview
In this paper, we take the first step to explore how we can leverage the prediction uncertainty to improve DNN-based Android malware detection models.
Our key insight is if we can identify uncertainty metrics that differ greatly between correct and incorrect predictions, we can use these metrics to pinpoint the potentially incorrectly-classified samples and correct their classification results accordingly

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 18.04. The codes depend on Python 3.6. Other packages (e.g., TensorFlow) can be found in the `./requirements.txt`.

## Configuration & Usage
#### 1. Datasets
* Three datasets are leveraged, namely that [Drebin](https://www.sec.cs.tu-bs.de/~danarp/drebin/), [VirusShare_Android_APK_2013](https://virusshare.com/) and [Androzoo](https://androzoo.uni.lu/). 
Note that for the security consideration, these three datasets are required to follow the policies of their own to obtain the Android applications. 

&emsp; For Drebin, we can download the malicious APKs from the official website and we provides sha256 codes of a portion of Drebin benign APKs, for which the corresponding APKs can be download from [Androzoo](https://androzoo.uni.lu/). 

&emsp; For Androzoo, we use the dataset built by researchers [Pendlebury et al.](https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury) All APKs can be downloaded from Androzoo.

&emsp; For Virusshare, we use the file named `VirusShare_Android_APK_2013.zip`. 

&emsp; For adversarial APKs, we resort to this [repository](https://github.com/deqangss/adv-dnn-ens-malware). 

* We additionally provide the preprocessed data files which are available at an anonymous [url](https://mega.nz/folder/bF8RQAAI#HeIhpUzDdqCdWdh4bAIZbg) (the size of unzip folder is ~213GB). 

#### 2. Configure
For the purpose of convenience, we provide a `conf` (Windows platform) / `conf-server` (Ubuntu) file to assist the customization (Please pick one and rename it `config` to use rather than both). Before running, all things are changed in the following:
* Modify the `project_root=/absolute/path/to/malware-uncertainty/`. 

* Modify the `database_dir=/absolute/path/to/datasets`. For more details (Optionally), there are `Drebin` or `Androzoo` malware datasets in this directory with the structure:
```
datasets
|---drebin
      |---malicious_samples  % malicious apps folder
      |---benign_samples     % benign apps foler
|---androzoo_tesseract
      |---malicious_samples
      |---benign_samples
      |   date_stamp.json    % date stamp for each app, we will provide
|---VirusShare_Android_APK_2013
      |---malicious_samples
      |---benign_samples
|---naive_data               % saving the preprocessed data files 
...
```
If no real apps are considered, the preprocessing data files make the project work as well. In this case, we need continue to configure the followings:
* Download the `datasets` from the anonymous [url](https://mega.nz/folder/bF8RQAAI#HeIhpUzDdqCdWdh4bAIZbg), and put the folder in the project root directory, namely `malware-uncertainty`. Please Note that this `datasets` is not necessary the same as the directory of `database_dir` in the second step. 
* Download the `naive_data` from the anonymous [url](https://mega.nz/folder/bF8RQAAI#HeIhpUzDdqCdWdh4bAIZbg), and put the folder in the `database_dir` directory, which is configured in the second step (need unzip, `mv naive_data.tar.gz database_dir; cd database_dir; tar -xvzf naive_data.tar.gz ./`).

#### 3. Usage
We suggest users to create a conda environment to run the codes. In this spirit, the following instructions may be helpful:
1. Create a new environment: `conda create -n mal-uncertainty python=3.6`
2. Activate the environment and install dependencies: `conda activate mal-uncertainty` and `pip install -r requirements.txt`
3. Next step:
* For training, all scripts are listed in `./run.sh`
* And then for producing figures and table data, the python code is `./experiments/table-figures.py` (we have not implemented this part for the malware detector `Droidetec`)

## Warning
* It is usually time consuming to perform feature extraction on Android applications.
* Two detectors (DeepDroid and Droidetec) are both RAM and computation consuming because the huge long sequence is used for promoting detection accuracy  


## License && Issues

We will make our codes public available under a formal license. For now, this is still an ongoing work and we plan to report more results in the future work. It is worth reminding that we found there two issues when checking our codes:
* No random seed set for friendly reproducing results exactly as the paper; nevertheless, the similar results can be achieved.
* The training, validation, and test datasets are split randomly, leading to a mess of results.

## Contact

Any questions, please do not hesitate to contact us (`Shouhuai Xu` email: `sxu@uccs.edu`, `Deqiang Li` email: `lideqiang@njust.edu.cn`)

