import torch
from torch.utils.data import Dataset, DataLoader
#import lava.lib.dl.slayer as slayer
#import lava.lib.dl.bootstrap as bootstrap
from snntorch import spikegen
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

import lava_dl.src.lava.lib.dl.slayer as slayer
import lava_dl.src.lava.lib.dl.bootstrap as bootstrap

from encodings import *
    
class SMILESDataset(Dataset):
    """Encoding the MoleculeACE dataset in such a way that it can be used as dataset for a SNN
    """
    def __init__(
        self, 
        path='datasets/CHEMBL234_Ki.csv',
        mode="train",
        val_size=0.2,
        transform=None,
        threshold=100,
        time_steps=1,
    ):
        """ Creates a FingerprintDataset.

        Parameters
        ----------
        path : str
            path of dataset as csv file with columns: smiles,exp_mean [nM],y,cliff_mol,split
        mode : str
            train/test flag
        transform : None or lambda or fx-ptr, optional
            transformation method to encode the smiles
        
        Attributes
        ----------
        """
        super(SMILESDataset, self).__init__()
        self.time_steps = time_steps
        self.path = path

        # read csv file
        self.df = pd.read_csv(path, dtype={'smiles':str})

        # pre-process labels
        self.df.loc[self.df['exp_mean [nM]'] < threshold, 'exp_mean [nM]'] = 1
        self.df.loc[self.df['exp_mean [nM]'] >= threshold, 'exp_mean [nM]'] = 0

        # Set right split
        if mode == "test":
            self.df = self.df.loc[self.df["split"] == 'test'].reset_index(inplace=False)
        if mode == "train":
            self.df = self.df.loc[self.df["split"] == 'train'].reset_index(inplace=False)

        # create dataset statistics
        self.targets = torch.tensor(self.df["exp_mean [nM]"].values, dtype=torch.int)
        self.num_active = self.df["exp_mean [nM]"].value_counts()[0]
        self.num_inactive = self.df["exp_mean [nM]"].value_counts()[1]

        # transform smiles
        self.df = transform(self.df, self.time_steps)
        
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Return a tuple consisting of the encoded smile at the requested index and its corresponding bioactivity. 
        
        Parameters
        ----------
        i: int 
            Index of smile in dataset

        Return
        ----------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple of (X,y) where X is the encoded smile in the csv at row i. X has shape CT, with C being the bit vector and T the time dimension. 
            y represents the label. In this case its binary. 
        """
        sample_row = self.df.iloc[i]
        sample_tensor = sample_row['smiles'].clone().detach()
        label = sample_row['exp_mean [nM]']
        return sample_tensor, int(label)

    def __len__(self) -> int:
        """Returns the number of molecules in the dataset.

        Returns
        -------
        int
            Number of molecules in the dataset with augmentation
        """
        return len(self.df)