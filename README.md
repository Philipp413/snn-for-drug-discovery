# Exploring the Chemical Universe with Spiking Neural Networks

This repository features the code used for my bachelor thesis called `Exploring the Chemical Universe with Spiking Neural Networks'. I designed a Spiking Neural Network in lava-dl for predicting bioactivity of molecules towards the dopamine D3 receptor, which can be run natively on the Intel Loihi 2. Furthermore, I evaluated the ability of SpikeGPT to learn the chemical language by benchmarking the validity, uniqueness and novelty of generated molecules. 

## Installation

You can install all necessary python dependencies using ```pip install -r requirements.txt``` if you are running on Linux with an x86_64 processor.

## De novo design

The code is based on the code of the [SpikeGPT](https://github.com/ridgerchu/SpikeGPT) and [S4 for de novo design](https://github.com/molML/s4-for-de-novo-drug-design). You can use the ```train.py``` to train a single SpikeGPT model and monitor the training in real time using [Weights and Biases](https://wandb.ai/site). Afterwards, use ```generate.py``` to benchmark validity, uniqueness and novelty, as well as grouping the errors based on categories. Hyperparameter tuning on multiple GPUs in parallel using raytune can be done with ```hyperparameter.py```. The hyperparameter tuning can be monitored in real time from any device using Weights and Biases.

## Virtual Screening

Hyperparameter search using Optuna for the ANN or SNN can be done using ```hyperparameter_ann.py``` or ```hyperparameter_snn.py``` respectively. Afterwards, the models can be benchmarked by running ```test_ann.py``` and ```test_snn.py```. The SNN was run on the Intel Loihi 2 using ```snn_inference.py```.