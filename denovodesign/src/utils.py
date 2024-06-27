########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os

try:
    NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
except:
    NUM_GPUS = 1

import json
import re
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data, ctx_len, epoch_length_fixed,name):
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed
        self.data = data
        self.name = name

        if 'MMapIndexedDataset' in str(type(self.data)):
            # self.vocab_size = 50257
            self.vocab_size = 50277
            print('current vocab size =', self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data._bin_buffer) // 2
            print(f'data has {self.data_size} tokens.')
        elif 'numpy' in str(type(self.data)):
            self.vocab_size = int(os.environ['VOCAB_SIZE'])
            print('current vocab size =', self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            print(f'data has {self.data_size} tokens.')
        else:
            # input: tokenised smile list, each sample is 100 tokens, each token separated by white space
            print('building token list...', end=' ')
            
            all_tokens = [item for row in self.data for item in row.split()]

            # compute vocabulary size as 
            unique = sorted(list(set(all_tokens)))
            self.vocab_size = len(unique)

            # total number of tokens in dataset
            self.data_size = len(all_tokens)

            print('data has %d tokens, %d unique. Including <BEG>, <END> and <PAD>' % (self.data_size, self.vocab_size))
            
            self.stoi = {ch: i for i, ch in enumerate(unique)} # A dictionary mapping unique characters to their index.
            self.itos = {i: ch for i, ch in enumerate(unique)} # A dictionary mapping indices to unique characters

            # create vocab file

            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f'./data/{self.name}_vocab.json', "w", encoding='utf-8') as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))

    def __len__(self):
        return self.epoch_length_fixed // NUM_GPUS

    def __getitem__(self, idx):
        curr_smile = self.data[idx]

        dix = [self.stoi[s] for s in curr_smile.split()]

        # does this perform the binary encoding?
        X = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return X, y


class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] == WORD_NAME[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
            self.vocab_size = len(self.tokenizer)
        else:
            self.charMode = True
            with open(WORD_NAME + '.json', "r", encoding="utf-8") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
        # out[self.UNKNOWN_CHAR] = -float('Inf')

        lastChar = int(x[-1]) # or make decisions based on last token?

        probs = F.softmax(torch.tensor(out), dim=-1)

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
        else:
            top_p = top_p_usual

        sorted_probs, s_index = torch.sort(probs, descending=True)

        # for j in range(30):
        #     pp = sorted_probs[j].item()
        #     if pp < 0.005:
        #         break
        #     ss = self.itos[int(s_index[j])].replace('\n','_')
        #     print(f'{math.floor(pp*100):>3.0f}{ss}', end='')
        # print('')

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

        probs[probs < cutoff] = 0
        # print("[" + str(round(cutoff,4)) + ' ' + str(round(to_float(sum(probs)),3)) + "]", end = "")

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return torch.multinomial(probs, num_samples=1)[0] # [0] returns index of sample


def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize_smile(smile, max_smile_length):
    # use REGEX to detect elements and SMILES syntax
    # \n gets filtered out by regex
    ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
    REGEX = (
        rf"(\[|\]|{ELEMENTS_STR}|"
        + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
    )
    RE_PATTERN = re.compile(REGEX)
    tokenized_smile = ['<BEG>'] + RE_PATTERN.findall(smile) + ['<END>']
    num_pad = abs(max_smile_length - len(tokenized_smile)) # num of padding tokens to add
    padded_list = tokenized_smile + ['<PAD>' for pad in range(0,num_pad)]
    return " ".join(padded_list) + '\n'
