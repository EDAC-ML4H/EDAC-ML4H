## EDAC-ML4H

This repository contains the source code for our findings paper titled "EDAC: Efficient 
Deployment of Audio Classification Models For COVID-19 Detection". The paper is submitted to 
<a href="https://ml4health.github.io/2023/index.html" target="blank_">Machine Learning for Health (ML4H)</a>.

### Abstract

The global spread of COVID-19 had severe consequences for public health and the world economy. 
The quick onset of the pandemic highlighted the potential benefits of cheap and deployable 
pre-screening methods to monitor the prevalence of the disease in a population. Various 
researchers made use of machine learning methods in an attempt to detect COVID-19. The solutions 
leverage various input features, such as CT scans or cough audio signals, with state-of-the-art 
results arising from deep neural network architectures. However, larger models require more 
compute; a pertinent consideration when deploying to the edge. To address this, we first 
recreated two models that use cough audio recordings to detect COVID-19. Through applying 
network pruning and quantisation, we were able to compress these two architectures without 
reducing the model's predictive performance.  Specifically, we were able to achieve an
$\sim105.76\times$ and an $\sim19.34\times$ reduction in the compressed model file size with 
corresponding  $\sim1.37\times$ and $\sim1.71\times$ reductions in the inference times of the 
two models. 

### Baseline models

| Model | Paper | GitHub | Data |
| --- | --- | --- | --- |
| Brogrammers | <a href="https://arxiv.org/abs/2110.06123" target="blank_"> link</a> | <a href="https://github.com/Saranga7/covid19--cough-diagnosis" target="blank_">link</a> | <a href="https://github.com/iiscleap/Coswara-Data" target="blank_">link</a> |
| Attention | <a href="https://link.springer.com/article/10.1007/s10844-022-00707-7" target="blank_">link</a> | <a href="https://github.com/skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram" target="blank_">link</a> | <a href="https://drive.google.com/file/d/1-4OIJsjky3PS7HRMdjQBibXLBwrQUVwH/view?usp=drive_link" target="blank_">link</a> |

### Source code

The source code provided in this repository can be used to generate the necessary data sets, 
train the baseline models and run the experiments. The data preprocessing steps and baseline 
model reproduction were aided by the GitHub repositories of the original papers.

The [notebooks](./notebooks/) folder contains all the Jupyter Notebooks used for the data 
generation and feature extraction to train the baseline models. These notebooks, including the 
one used for the Brogrammers baseline training, can be run locally within reasonable time. A 
notebook dedicated for plot and table generation for the paper is also provided.

The [scripts](./scripts/) folder contains the [spec_augment_sets.py](./scripts/spec_augment_sets.py) used for the CoughVid data augmentation. While it can be run locally, it is expected to take over an hour to run. The remaining scripts are used for the Attention model baseline training and running the experiments. These tasks are significantly more computationally intensive, therefore they were done on <a href="https://www.kaggle.com/" target="blank_">Kaggle</a>.

### Running the experiments

The experiments are fully scripted and can be run on Kaggle using 2x GPU T4 for acceleration and the preprocessed datasets.

**Note**: To run the experiments, make sure to install the <a href="https://www.tensorflow.org/model_optimization/api_docs/python/tfmot" target="blank_">Tensorflow Model Optimization</a> package in the Kaggle environment.

#### Brogrammers

The Brogrammers experiments can be run using the corresponding Python script on Kaggle for 20 random seeds. The same script can be used for the Constant Sparsity and the Polynomial Decay experiments. To switch between the experiments, edit <a href="https://github.com/EDAC-ML4H/EDAC-ML4H/blob/5fc22b06b6d4f35155142c6239ed16aeeffda55e/scripts/brogrammers_workshop_experiments.py#L295C5-L295C16" target="blank_">line 295</a> in the provided [script](./scripts/brogrammers_workshop_experiments.py). Set the `experiment` variable to `const` to run the Constant Sparsity experiment or set it to `poly` to run the polynomial experiment. Each experiment run should take just about 4 hours. The results of the experiments can be downloaded once the scripts finish.

**Note**: The output of each experiment run contains the saved TFLite model files and a JSON file for each run with the collected results of the run. This results in an output of more than 1000 files. To download them, save them as a dataset from the output of the script, otherwise the download will fail with a network error.

#### Attention

The Attention experiments are set up with a similar [script](./scripts/attention_workshop_experiments.py). However, each run of this script only corresponds to 1 run for 1 experiment and takes about 7 hours to run. The script can be rerun multiple times to generate outputs for different random seeds and experiments. Edit <a href="https://github.com/EDAC-ML4H/EDAC-ML4H/blob/5fc22b06b6d4f35155142c6239ed16aeeffda55e/scripts/attention_workshop_experiments.py#L340" target="blank_">line 340</a> to set the experiment the same way as for the Brogrammers script and <a href="https://github.com/EDAC-ML4H/EDAC-ML4H/blob/5fc22b06b6d4f35155142c6239ed16aeeffda55e/scripts/attention_workshop_experiments.py#L345" target="blank_">line 345</a> to set the run starting from 0. For our purposes, we did 5 runs for each experiments, totalling at just about 70 hours of GPU time.

**Note**: Given the separate runs, the results can be directly downloaded and collected after the runs. As for the Brogrammers model, the results contain the saved TFLite model files and a JSON file with the collected results of the run.

### Processed data availability

The preprocessed and split data sets for both models, along with the baseline model weights, will be made available on Kaggle on acceptance of the paper.

### Reference

If you found our paper and the content of this repository valuable, please cite our paper as follows.

```
Will be added after acceptance.
```

### Licence

The content of this repository can be freely used under the [Apache-2.0 licence](./LICENCE).

### Contact

*Will be added after acceptance.*