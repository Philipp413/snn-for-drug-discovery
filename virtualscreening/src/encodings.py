import torch
from rdkit.Chem import MACCSkeys
from rdkit import Chem
import pandas as pd
from snntorch import spikegen
from rdkit.Chem import AllChem

def macc167xT(df,time):
    new_df = df.copy()
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: torch.tensor(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile)),dtype=torch.float))
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: spikegen.rate(smile, time).permute(1,0))
    return new_df

def macc167(df,time):
    new_df = df.copy()
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: torch.tensor(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile)),dtype=torch.float))
    return new_df

def morgan1024xT(df,time):
    new_df = df.copy()
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile),2,nBits=1024,)))
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: torch.tensor(smile,dtype=torch.float))
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: spikegen.rate(smile, time).permute(1,0))
    return new_df

def morgan1024(df,time):
    new_df = df.copy()
    new_df['smiles'] = new_df['smiles'].apply(lambda smile: torch.tensor(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile),2,nBits=1024),dtype=torch.float))
    return new_df