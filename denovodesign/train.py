import numpy as np
import os
import json
import torch
import timeit
import argparse
from src.utils import TOKENIZER
from src.model import GPT, GPTConfig
from src.utils import Dataset
from src.torch_callbacks import EarlyStopping
from src.spikeGPT import SpikeGPT
import matplotlib.ticker as ticker
import wandb
from zeus.monitor import ZeusMonitor
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.manual_seed(42)

api_key = None # TODO: Put your wandb API key here

wandb.login(key=api_key)

parser = argparse.ArgumentParser("SpikeGPT training")
parser.add_argument("--model_id", help="ID of the model to be trained", type=int)
args = parser.parse_args()
print(f'Model ID: {args.model_id}')

config= {
    "max_epochs": 100,
    "emb_size" : 256,
    "batch_size": 1024,
    "n_layers" : 8,
    "learning_rate": 0.000002271,
    "dropout_rate" : 0.1,
    "ctx_len" : 100,
    "temperature" : 1,
    "num_designs" : 1024,
    "gen_batch_size" : 128,
    "betas" : (0.9,0.999),
    "eps" : 1e-08,
}

project_name = "spike-gpt"

run = wandb.init(project=project_name,name=f"experiment{args.model_id}",config=config,dir="./wandb/")

monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

print('Initialize model...')

spikeGPT = SpikeGPT(
    n_max_epochs=config["max_epochs"],
    batch_size=config["batch_size"], 
    device="cuda",  
    n_layers=config["n_layers"],
    learning_rate_init=config["learning_rate"],
    learning_rate_final=config["learning_rate"],
    dropout_rate=config["dropout_rate"],
    ctx_len=config["ctx_len"],
    model_dim=config["emb_size"],
)

experiment_dir = f'./experiments/experiment{args.model_id}/'
os.makedirs(experiment_dir, exist_ok=True)

wandb.watch(spikeGPT.spikeGPT_model)

print('Start training...')

training_dir = f'./datasets/ChEMBL_train.txt'
val_dir = f"./datasets/ChEMBL_valid.txt"

# Pretrain the model on ChEMBL
monitor.begin_window("total_train")
history = spikeGPT.train(
    training_molecules_path=training_dir,
    val_molecules_path=val_dir,
    callbacks=[
        EarlyStopping(
            patience=5, delta=1e-5, criterion="val_loss", mode="min"
        ),
    ]
)
mes = monitor.end_window("total_train")

total_time_train = mes.time # in s
total_energy_train = mes.total_energy # in J

spikeGPT.save(experiment_dir, total_time_train,total_energy_train) # Save the model