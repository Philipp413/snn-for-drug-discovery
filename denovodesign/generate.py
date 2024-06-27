import numpy as np
import math, os, sys, types, time, gc,json
import torch
import timeit
import argparse
from matplotlib import pyplot as plt
from src.utils import TOKENIZER
from src.model import GPT, GPTConfig
from src.utils import Dataset, tokenize_smile
from src.torch_callbacks import EarlyStopping
from src.spikeGPT import SpikeGPT
import matplotlib.ticker as ticker
import wandb
import rdkit
from io import StringIO
from rdkit import Chem
from rdkit import Chem, rdBase
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

rdBase.LogToPythonStderr()

torch.manual_seed(42)

def benchmark_smiles(design_dir,train_dir):
    with open(design_dir, "r") as f:
        benchmark_smiles = [line.strip() for line in f.readlines() if line.strip()]

    with open(train_dir, "r") as f:
        training_smiles = [line.strip() for line in f.readlines()] 

    # validity - the number (and frequency) of SMILES corresponding to chemically valid molecules. A molecule is considered valid if Chem.MolFromSmiles(smiles) does not return null. 
    valid_designs = [smile for smile in benchmark_smiles if Chem.MolFromSmiles(smile) is not None] # use same as S4
    caconicalized = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),canonical=True) for smi in valid_designs]
    print(f'{len(valid_designs)-len(caconicalized)} smiles filtered out by caconicalization')
    print(f'Number of valid designs: {len(valid_designs)}')
    validity = len(valid_designs)/len(benchmark_smiles)
    print(f'Validity: {validity}')

    # # novelty - the number (and frequency) of unique and valid designs that are not included in the training set. 
    unique_canonicalized = list(set(caconicalized))
    training_smiles_set = set(training_smiles)
    novel_designs = [smile for smile in unique_canonicalized if smile not in training_smiles_set]
    novelty = len(novel_designs)/len(benchmark_smiles)
    print(f'Novelty: {novelty}')

    # # first caconicalization and filter out invalid designs
    uniqueness = len(unique_canonicalized)/len(benchmark_smiles)
    print(f'Uniqueness: {uniqueness}')

    failures=[]
    sio = sys.stderr = StringIO()
    for m in benchmark_smiles:
        if Chem.MolFromSmiles(m) is None:
            failures.append(sio.getvalue())
            sio = sys.stderr = StringIO() # reset the error logger
    sys.stderr = sys.__stderr__
    
    invalid_categories = {
        "Unclosed Rings" : 0,
        "Kekulization" : 0,
        "Extra Open Parentheses" : 0,
        "Extra Close Parentheses" : 0,
        "Valence" : 0,
        "Syntax Errors" : 0,
        "Aromaticity" : 0,
        "Self Bond" : 0,
        "Ring Duplicate Bond" : 0,
    }
    ERROR_TYPE_TO_CATEGORY = {
        "Unclosed Rings": "rings",
        "Kekulization": "bond assignment",
        "Extra Open Parentheses": "branching",
        "Extra Close Parentheses": "branching",
        "Valence": "bond assignment",
        "Syntax Errors": "misc",
        "Aromaticity": "bond assignment",
        "Self Bond": "rings",
        "Ring Duplicate Bond": "rings",
    }

    for smile in failures:
        if "Unkekulized atoms" in smile:
            invalid_categories["Kekulization"] +=1
        if "unclosed ring for input" in smile:
            invalid_categories["Unclosed Rings"] += 1
        if "extra open parentheses" in smile:
            invalid_categories["Extra Open Parentheses"] += 1
        if "extra close parentheses" in smile:
            invalid_categories["Extra Close Parentheses"] += 1
        if "syntax error" in smile or "Failed parsing SMILES" in smile:
            invalid_categories["Syntax Errors"] += 1
        if "Explicit valence in atom" in smile:
            invalid_categories["Valence"] += 1
        if "non-ring atom" in smile:
            invalid_categories["Aromaticity"] += 1
        if "itself" in smile and "duplicated ring closure" in smile:
            invalid_categories["Self Bond"] += 1
        if "duplicates bond between atom" in smile:
            invalid_categories["Ring Duplicate Bond"] += 1

    print(invalid_categories)

    error_types_counts = {error_type: 0 for error_type in ["rings", "bond assignment", "branching", "misc"]}

    # Aggregate counts based on error type
    for category, count in invalid_categories.items():
        error_type = ERROR_TYPE_TO_CATEGORY.get(category)
        if error_type:
            error_types_counts[error_type] += count
    error_types_counts
    
    total_error = len(failures)
    
    return validity, novelty, uniqueness, invalid_categories, error_types_counts, total_error

parser = argparse.ArgumentParser("SpikeGPT generation")
parser.add_argument("--model_id", help="ID of the model to be trained", type=int)
args = parser.parse_args()
print(f'Model ID: {args.model_id}')

config = {
    "num_designs" : 10240,
    "gen_batch_size" : 2048,
    "temperature" : 1.0
}

experiment_dir = f'./experiments/experiment{args.model_id}/'

generations_dir = f'{experiment_dir}generations/'
os.makedirs(generations_dir, exist_ok=True)
generations_file = f'{generations_dir}designs_{args.model_id}.smiles'
training_dir = f'./datasets/ChEMBL_train.txt'

print('Initialize model...')

spikeGPT = SpikeGPT.from_file(experiment_dir)

print("Start generation...")

designs, lls = spikeGPT.design_molecules(n_designs=config["num_designs"], batch_size=config["gen_batch_size"], temperature=config["temperature"])

with open(generations_file, "w") as f:
     f.write("\n".join(designs))

print("Benchmarking designs")

validity, novelty, uniqueness, invalid_categories, error_types_counts, total_error = benchmark_smiles(generations_file, training_dir)

results = {"validity": validity, 
                "novelty": novelty, 
                "uniqueness":uniqueness, 
                "temperature": config["temperature"],
                "num_designs": config["num_designs"],
                "gen_batch_size": config["gen_batch_size"],
                "Errors": invalid_categories, 
                "Errors grouped": error_types_counts,
                "total_error" : total_error}

with open(f'{experiment_dir}benchmark_{config["temperature"]}.json', "w") as f:
    print("saving results")
    json.dump(results, f, indent=4)