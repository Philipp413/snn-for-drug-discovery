import wandb
import random
from ray import tune, train, air
from io import StringIO
import ray
import numpy as np
import math, os, sys, types, time, gc, json
import torch
import timeit
import argparse
from src.utils import TOKENIZER
from src.model import GPT, GPTConfig
from src.utils import Dataset, tokenize_smile
from src.torch_callbacks import EarlyStopping
from src.spikeGPT import SpikeGPT
import matplotlib.ticker as ticker
import rdkit
from pathlib import Path
from rdkit import Chem
from rdkit import Chem, rdBase
import argparse

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

rdBase.LogToPythonStderr() # try version 2023

torch.manual_seed(42)

NUM_DESIGNS = 10240
BATCH_SIZE = 1024 # for generation
TEMPERATURE = 1.0

def benchmark_smiles(gen, train, temperature,model_dir):
    with open(gen, "r") as f:
        benchmark_smiles = [line.strip() for line in f.readlines()]

    with open(train, "r") as f:
        training_smiles = [line.strip() for line in f.readlines()] 

    # validity - the number (and frequency) of SMILES corresponding to chemically valid molecules. A molecule is considered valid if Chem.MolFromSmiles(smiles) does not return null. 
    valid_designs = [smile for smile in benchmark_smiles if Chem.MolFromSmiles(smile) is not None] # use same as S4
    caconicalized = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),canonical=True) for smi in valid_designs]
    print(f'{len(valid_designs)-len(caconicalized)} smiles filtered out by caconicalization')
    print(f'Number of valid designs: {len(valid_designs)}')
    validity = len(valid_designs)/len(benchmark_smiles)
    print(f'Validity: {validity}')

    # novelty - the number (and frequency) of unique and valid designs that are not included in the training set. 
    unique_canonicalized = list(set(caconicalized))
    training_smiles_set = set(training_smiles)
    novel_designs = [smile for smile in unique_canonicalized if smile not in training_smiles_set]
    novelty = len(novel_designs)/len(benchmark_smiles)
    print(f'Novelty: {novelty}')

    uniqueness = len(set(caconicalized))/len(benchmark_smiles)
    print(f'Uniqueness: {uniqueness}')
    
    results = {"validity": validity, "novelty": novelty, "uniqueness":uniqueness, "temperature": temperature}
    with open(f'{model_dir}benchmark_{temperature}.json', "w") as f:
        print("saving results")
        json.dump(results, f, indent=4)
    return validity, uniqueness, novelty

def train_model(config):
    tune.utils.wait_for_gpu()
    model_id=config["model_id"]
    run = wandb.init(project="Spike-GPT-HPO",
                     name=f"run{model_id}",
                     config=config,
                     dir="./wandb/")
    experiment_dir = f'./experiments/raytune/experiment{config["model_id"]}/'
    os.makedirs(experiment_dir, exist_ok=False)
    generations_dir = f'{experiment_dir}generations/'
    os.makedirs(generations_dir, exist_ok=True)
    generations_file = f'{generations_dir}designs_{config["model_id"]}.smiles'
        
    print('Initialize model...')

    spikeGPT = SpikeGPT(
        model_dim=config["embedding_size"],
        n_max_epochs=400,
        batch_size=config["batch_size"],  
        device="cuda",  
        n_layers=config["n_layers"],
        learning_rate_init=config["learning_rate"],
        learning_rate_final=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        ctx_len=config["ctx_len"], 
        eps=config["eps"],
        betas=config["betas"], 
        experiment_id=config["model_id"],
    )
    print('Start training...')
    training_dir="/vol2/BachelorThesis/SpikeGPT/data/S4_train.txt"

    history = spikeGPT.train(
        training_molecules_path=training_dir,
        val_molecules_path="/vol2/BachelorThesis/SpikeGPT/data/S4_valid.txt",
        callbacks=[
            EarlyStopping(
                patience=5, delta=1e-5, criterion="val_loss", mode="min"
            ),
        ]
    )
    
    with open(f'{experiment_dir}training_loss.json', "w") as f:
        print("saving results")
        json.dump(history["train_loss"], f, indent=4)
    with open(f'{experiment_dir}val_loss.json', "w") as f:
        print("saving results")
        json.dump(history["val_loss"], f, indent=4)
    
    spikeGPT.save(experiment_dir)
    
    designs, lls = spikeGPT.design_molecules(n_designs=NUM_DESIGNS, batch_size=BATCH_SIZE, temperature=TEMPERATURE)

    with open(generations_file, "w") as f:
         f.write("\n".join(designs))

    print('Benchmark generated molecules...')

    validity, uniqueness, novelty = benchmark_smiles(generations_dir, training_dir, TEMPERATURE,experiment_dir)
    
    wandb.log({"validity": validity,"uniqueness": uniqueness,"novelty": novelty})
                     
    return {"validity": validity,"uniqueness": uniqueness,"novelty": novelty}
    

if __name__ == "__main__":

    num_trials = 200
    
    api_key= None # TODO: Put your wandb API key here

    wandb.login(key=api_key)
    
    config = {
        "batch_size": tune.choice([512,1024]),
        "n_layers": tune.randint(2, 9),
        "learning_rate": tune.loguniform(1e-6, 1e-1), 
        "dropout_rate": tune.choice([0.0,0.1,0.2]),
        "embedding_size": tune.choice([32,64,128,256]),
        "model_id": tune.grid_search([i for i in range(num_trials)]),
        "betas" : (0.9,0.999),
        "eps" : 1e-08,
        "ctx_len" : 100,
    }

    gpus_per_trial = 1

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        run_config=train.RunConfig(
            name="spikeGPT",
            storage_path=Path("./raytune").resolve(),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False,
                checkpoint_frequency=0,
                num_to_keep=1,
            ),
            log_to_file=False,
            ),
        param_space=config,
    )
    results = tuner.fit()