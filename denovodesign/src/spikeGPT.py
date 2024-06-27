import json
import math
import os
import types
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from . import smiles_utils, torch_callbacks
from .dataloaders import create_dataloader

import numpy as np
import sys, types, time, gc

from .utils import TOKENIZER
from .model import GPT, GPTConfig
from .trainer import Trainer, TrainerConfig
from .utils import Dataset, tokenize_smile
from .model_run import RWKV_RNN
import matplotlib.ticker as ticker

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

class SpikeGPT:
    """The SpikeGPT model for de novo design."""

    def __init__(
        self,
        model_dim: int = 256,
        n_layers: int = 4,
        vocab_size: int = 37,
        sequence_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate_init: float = 0.001,
        learning_rate_final: float = 5e-4,
        batch_size: int = 2048,
        device: str = "cuda",
        model_type: str = "RWKV",
        ctx_len: int = 256,
        betas = (0.9,0.9),
        eps = 4e-9,
        num_workers = 0,
        epoch_length_fixed: int = 10000, # length of each mini-epoch
        epoch_length_valid_fixed: int = 1000,
        grad_norm_clip = 1.0,
        warmup_tokens = 0,
        epoch_save_frequency = 10,
        dropout_rate=0.03,
        total_training_time=0,
        total_training_energy=0,
        experiment_id=0,
    ) -> None:
        """Creates an `SpikeGPT` instance.

        Parameters
        ----------
        model_dim : int
            The size of each embedding vector.
        n_layers : int
            The number of SpikeGPT blocks in the model.
        dropout : float
            The dropout rate.
        vocab_size : int
            The size of the vocabulary.
        sequence_length : int
            The length of the sequences.
        n_max_epochs : int
            The maximum number of epochs to train for.
        learning_rate : float
            The learning rate.
        batch_size : int
            The batch size.
        device : str
            The device to put the model on, *e.g.,* `"cuda"` or `"cpu"`.
        betas : tuple[float,float]
            Beta coefficients
        eps : float
            Epsilon value
        """

        self.model_dim = model_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.n_max_epochs = n_max_epochs
        self.learning_rate_init = learning_rate_init
        self.learning_rate_final = learning_rate_final
        self.batch_size = batch_size
        self.device = device
        self.model_type = model_type
        self.ctx_len = ctx_len
        self.betas = betas
        self.eps = eps
        self.num_workers = num_workers
        self.epoch_length_fixed = epoch_length_fixed
        self.epoch_length_valid_fixed = epoch_length_valid_fixed
        self.grad_norm_clip = grad_norm_clip
        self.warmup_tokens = warmup_tokens
        self.epoch_save_frequency = epoch_save_frequency
        self.dropout_rate = dropout_rate
        self.total_training_time = total_training_time
        self.total_training_energy=total_training_energy
        self.experiment_id=experiment_id

        # These are set during training
        self.token2label = None
        self.label2token = None

        self.spikeGPT_model = GPT(GPTConfig(self.vocab_size, self.ctx_len, model_type=self.model_type,n_layer=self.n_layers,n_embd=self.model_dim,dropout_rate=self.dropout_rate)).cuda()
        
    @classmethod
    def from_file(cls, loaddir: str):
        """Loads an `SpikeGPT` instance from a directory.

        Parameters
        ----------
        loaddir : str
            The directory to load the model from.

        Returns
        -------
        SpikeGPT
            The loaded model.
        """
        with open(f"{loaddir}/init_arguments.json", "r") as f:
            properties = json.load(f)
        
        spikeGPT_model = GPT(GPTConfig(
            properties["vocab_size"], 
            properties["ctx_len"],
            model_type=properties["model_type"],
            n_layer=properties["n_layers"],
            n_embd=properties["model_dim"],
            dropout_rate=properties["dropout_rate"])).cuda()
        
        m2 = torch.load(f"{loaddir}/model.pt")
        spikeGPT_model.load_state_dict(m2)
        
        token2label = properties.pop("token2label")
        label2token = properties.pop("label2token")
        instance = cls(**properties)
        instance.device = "cpu"
        instance.spikeGPT_model = spikeGPT_model
        instance.spikeGPT_model.to(instance.device)
        instance.token2label = token2label
        instance.label2token = {
            int(label): token for label, token in label2token.items()
        }
        return instance

    def train(
        self,
        training_molecules_path: str,
        val_molecules_path: str,
        callbacks: List[torch_callbacks.TorchCallback] = None,
    ) -> Dict[str, List[float]]:
        """Trains the model. The inputs are the paths to the training and validation molecules.
        The paths should point either to a .txt file that contains one SMILES per line, or to a zip file with the same structure.
        The optional callbacks can be used to monitor or configure training.
        The training history is returned as a dictionary.

        Parameters
        ----------
        training_molecules_path : str
            The path to the training molecules. Can be a zip file or a text file. Must contain one SMILES string per line.
        val_molecules_path : str
            The path to the validation molecules. Must have the same structure as `training_molecules_path`.
        callbacks : List[torch_callbacks.TorchCallback], optional
            A list of callbacks to use during training. See the documentation of the `torch_callbacks` module for available options.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the training history. The keys are `train_loss` and `val_loss` and the values are lists of the metric values at each epoch.
        """
        self.spikeGPT_model = self.spikeGPT_model.to(self.device)
        train_dataloader = create_dataloader(
            training_molecules_path,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length + 1, 
            num_workers=8,
            shuffle=True,
            token2label=self.token2label,
        )
        self.token2label = train_dataloader.dataset.token2label
        
        # uncomment to determine vocab size
        #with open(f"./token2label.json", "w") as f:
        #    json.dump(self.token2label, f, indent=4)
        
        self.label2token = {v: k for k, v in self.token2label.items()}

        val_dataloader = create_dataloader(
            val_molecules_path,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length + 1,
            num_workers=8,
            shuffle=True,
            token2label=self.token2label,
        )
        
        test_dataset = None
        t1 = self.n_max_epochs
        t2 = self.epoch_length_fixed
        t3 = self.ctx_len
        print(self.n_max_epochs)
        print(self.epoch_length_fixed)
        print(self.ctx_len)
        final_tokens = t1 * t2 * t3
        print(final_tokens)
        print(f"Batch size: {self.batch_size}")
        
        tconf = TrainerConfig(model_type=self.model_type,
                              max_epochs=self.n_max_epochs, 
                              batch_size=self.batch_size,
                              learning_rate=self.learning_rate_init, 
                              lr_decay=False, 
                              lr_final=self.learning_rate_final, 
                              betas=self.betas, 
                              eps=self.eps, 
                              grad_norm_clip=self.grad_norm_clip,
                              warmup_tokens=self.warmup_tokens, 
                              final_tokens=final_tokens, 
                              num_workers=self.num_workers, 
                              epoch_save_frequency=self.epoch_save_frequency, 
                              epoch_save_path="model/")
        # TODO: understand how final_tokens work and set properly
        trainer = Trainer(self.spikeGPT_model, train_dataloader, val_dataloader, test_dataset, tconf)
        
        self.n_max_epochs, history = trainer.train(callbacks=callbacks)
        
        return history

    @torch.no_grad()
    def design_molecules(
        self,
        n_designs: int,
        batch_size: int,
        temperature: float,
        model_path: str,
    ) -> Tuple[List[str], List[float]]:
        """Designs molecules using the trained model. The number of designs to generate is specified by `n_designs`.
        The designs are generated in batches of size `batch_size`. The temperature is used to control the diversity of the generated designs.
        The designs and their log-likelihoods are returned as a tuple.

        Parameters
        ----------
        n_designs : int
            The number of designs to generate.
        batch_size : int
            The batch size to use during generation.
        temperature : float
            The temperature to use during generation.

        Returns
        -------
        Tuple[List[str], List[float]]
            A tuple containing the generated SMILES strings and their log-likelihoods.
        """
        if self.token2label is None or self.label2token is None:
            raise ValueError("This model is untrained.")
        
        args = types.SimpleNamespace()
        
        args.RUN_DEVICE = "cuda" # 'cuda' // 'cpu' (already fast)
        args.FLOAT_MODE = "fp32" 
        
        args.MODEL_NAME = model_path
        args.n_layer = self.n_layers
        args.n_embd = self.model_dim
        args.ctx_len = self.ctx_len
        args.vocab_size = self.vocab_size
        args.head_qk = 0
        args.pre_ffn = 0
        args.grad_cp = 0
        args.my_pos_emb = 0
        
        self.gen_model = RWKV_RNN(args)
            
        self.gen_model = self.gen_model.to(self.device)
        self.gen_model.eval()

        n_batches = math.ceil(n_designs / batch_size)
        designs, likelihoods = list(), list()
        for batch_idx in range(n_batches):

            state = None
            mem1 = None
            mem2 = None

            if batch_idx == n_batches - 1:
                batch_size = n_designs - batch_idx * batch_size
            X_test = (
                torch.zeros(batch_size, 1).to(torch.int) + self.token2label["[BEG]"]
            )
            X_test = X_test.to(self.device)

            batch_designs, batch_likelihoods = list(), list()
            for __ in range(self.sequence_length):
                preds, state, mem1, mem2 = self.gen_model.forward(X_test,state,mem1,mem2)
                preds = preds.view(batch_size,self.vocab_size)
                softmax_preds = F.softmax(preds, dim=-1).detach().cpu().numpy().tolist()
                preds = preds.detach().cpu().numpy().tolist()
                token_labels, token_likelihoods = list(), list()
                for pred_idx, pred in enumerate(preds):
                    pred_temperature = np.exp(np.array(pred) / temperature)
                    pred_sum = sum(pred_temperature) 
                    pred_normed = [p / pred_sum for p in pred_temperature]
                    probas = np.random.multinomial(1, pred_normed)
                    token_label = np.argmax(probas)
                    token_labels.append(token_label)
                    token_likelihood = softmax_preds[pred_idx][token_label]
                    token_likelihoods.append(token_likelihood)

                batch_designs.append(token_labels)
                batch_likelihoods.append(token_likelihoods)
                
                X_test = torch.tensor(token_labels).unsqueeze(1).to(self.device) 

            designs.append(np.array(batch_designs).T)
            likelihoods.append(np.array(batch_likelihoods).T)
            print(f'finished batch: {batch_idx}', end='\r')
        designs = np.concatenate(designs, axis=0).tolist()
        
        molecules = [
            [
                self.label2token[label]
                for label in design
                if self.label2token[label] not in ["[BEG]", "[END]", "[PAD]"]
            ]
            for design in designs
        ]
        molecule_lens = [
            len(molecule) + 2 for molecule in molecules
        ]  # +2 for [BEG] and [END] token
        smiles = ["".join(molecule) for molecule in molecules]
        loglikelihoods = np.log(np.concatenate(likelihoods, axis=0)).tolist()
        mean_loglikelihoods = [
            np.mean(ll[: mol_len - 1])
            for ll, mol_len in zip(loglikelihoods, molecule_lens)
        ]

        return smiles, mean_loglikelihoods

    def save(self, path: str, total_training_time: float = 0.0,total_training_energy = 0.0):
        """Saves the model to a directory. The directory will be created if it does not exist.

        Parameters
        ----------
        path : str
            The directory to save the model to.
        """
        print("Saving model to", path)
        os.makedirs(path, exist_ok=True)
        torch.save(self.spikeGPT_model.state_dict(), f"{path}/model.pt")
        properties = {p: v for p, v in self.__dict__.items() if p != "spikeGPT_model"}
        properties["total_training_time"] = total_training_time
        properties["total_training_energy"] = total_training_energy
        
        with open(f"{path}/init_arguments.json", "w") as f:
            json.dump(properties, f, indent=4)
