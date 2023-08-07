# MALCERTAIN


This code repository our ICSE 0204 paper titled **MALCERTAIN:Enhancing Deep Neural Network Based Android Malware Detection by Tackling Prediction Uncertainty**.
 
## Overview
In this paper, we take the first step to explore how we can leverage the prediction uncertainty to improve DNN-based Android malware detection models.
Our key insight is if we can identify uncertainty metrics that differ greatly between correct and incorrect predictions, we can use these metrics to pinpoint the potentially incorrectly-classified samples and correct their classification results accordingly

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 20.04. The codes depend on Python 3.8.10. Other packages (e.g., TensorFlow) can be found in the `./requirements.txt`.

##  Usage
#### 1. train base model and uncertainty estimation models
for example: 
     train deepdrebin base model: /myexperiment/train_uncertainity_model/Vanilla.py

     python Vanilla.py -train_type drebin -model_type small   ## The parameter "small" indicates that the base model is trained with a small training set. 

     python Bayesian.py -train_type drebin -model_type small ##train bayesian DNN model to estimation uncertainty
The other models are trained in a similar way.


#### 2. Calculation of uncertainty metrics

get uncertainty metrics: myexperiment/uncertainity_metrics_utils/main_uc_metrics.py

     python main_uc_metrics.py -model_arch drebin -model_type small -data_type ood

Get all the metrics and save them separately as csv files

#### 3. train correction model and correction

train：myexperiment/uncertainity_metrics_utils/ml_true_flase.py

      python ml_true_flase.py -experiment_type train -save_model y -data_type small_drebin
      
      ###  experiment_type: Type of experiment,training correction model or resultant correction.
      ###  save_model: Whether to save trained correction models
      ###  data_type: Data types for training corrective models，(small_drebin,small_multi)
      ###  train_data_size: The effect of the scale of the training data on the corrective model,[1.0,0.8,0.4,0.2,0.1]
      ###  banlance: The effect of whether the training data is balanced or not on the corrective model



 

  



