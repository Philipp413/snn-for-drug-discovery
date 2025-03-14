
import types
import torch
import math, os, gc
from torch.nn import functional as F
import torch.nn as nn
from typing import List, Dict
from src.spikingjelly.clock_driven import neuron,functional,surrogate
from src.spikingjelly.clock_driven.surrogate import ATan

MyModule = nn.Module
def __nop(ob):
    return ob
MyFunction = __nop

# # try torchdynamo
# import torchdynamo
# MyFunction = torchdynamo.optimize(os.environ["RWKV_RUN_BACKEND"]) # !!!BUGGY!!! wrong output

# try torch jit --> faster for fp32, slower for fp16 (why?)
# if os.environ["RWKV_JIT_ON"] == "1":
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method

RWKV_HEAD_QK_DIM = 0
#print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM} RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]}\n')

DEBUG_TIME = False   # True False - show trained time-coeffs

RWKV_RESCALE_LAYER = 6 # set x=x/2 every X layer

############################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.RUN_DEVICE = args.RUN_DEVICE
        with torch.no_grad():
            w = torch.load(args.MODEL_NAME + 'model.pt') # ,map_location='cpu' 
            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(args.ctx_len+1, -1)[:-1,:]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        print(x, w[x].numpy())
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    if self.FLOAT_MODE == "fp32":
                        w[x] = w[x].float()
                    elif self.FLOAT_MODE == "bf16":
                        w[x] = w[x].bfloat16()
                    elif self.FLOAT_MODE == "fp16":
                        w[x] = w[x].half()

                w[x].requires_grad = False
                if args.RUN_DEVICE == 'cuda' and x != 'emb.weight':
                    w[x] = w[x].cuda()

                if ('blocks.' not in x) or ('blocks.0.' in x):
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)
    
    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw, mem):
        batch_size, _ = x.size()

        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + state[:, 5*i+0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + state[:, 5*i+0].type(torch.bfloat16) * (1 - time_mix_r)
            state[:, 5*i+0] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + state[:, 5*i+0].half() * (1 - time_mix_k)
            xr = x * time_mix_r + state[:, 5*i+0].half() * (1 - time_mix_r)
            state[:, 5*i+0] = x.float()
        else:
            xk = x * time_mix_k + state[:, 5*i+0] * (1 - time_mix_k)
            xr = x * time_mix_r + state[:, 5*i+0] * (1 - time_mix_r)
            state[:, 5*i+0] = x

        r = torch.sigmoid((rw @ xr.T).T)  # Transpose for batch matrix multiplication
        k = torch.square(torch.relu((kw @ xk.T).T))
        kv = (vw @ k.T).T

        output = torch.empty_like(kv)
        for batch_idx in range(batch_size):
            output[batch_idx] = mem[i][batch_idx](r[batch_idx] * kv[batch_idx])

        return output


    @MyFunction
    def SA(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow, mem):
        batch_size, _ = x.size()

        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + state[:, 5*i+1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + state[:, 5*i+1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + state[:, 5*i+1].type(torch.bfloat16) * (1 - time_mix_r)
            state[:, 5*i+1] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + state[:, 5*i+1].half() * (1 - time_mix_k)
            xv = x * time_mix_v + state[:, 5*i+1].half() * (1 - time_mix_v)
            xr = x * time_mix_r + state[:, 5*i+1].half() * (1 - time_mix_r)
            state[:, 5*i+1] = x.float()
        else:
            xk = x * time_mix_k + state[:, 5*i+1] * (1 - time_mix_k)
            xv = x * time_mix_v + state[:, 5*i+1] * (1 - time_mix_v)
            xr = x * time_mix_r + state[:, 5*i+1] * (1 - time_mix_r)
            state[:, 5*i+1] = x

        r = torch.sigmoid(rw @ xr.T).T  # Transpose for batch matrix multiplication
        k = (kw @ xk.T).T
        v = (vw @ xv.T).T

        if '16' in self.FLOAT_MODE:
            kk = k.float()
            vv = v.float()
        else:
            kk = k
            vv = v

        aa = state[:, 5*i+2]
        bb = state[:, 5*i+3]
        pp = state[:, 5*i+4]
        ww = time_first + kk

        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        a = e1 * aa + e2 * vv
        b = e1 * bb + e2

        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)

        state[:, 5*i+2] = e1 * aa + e2 * vv
        state[:, 5*i+3] = e1 * bb + e2
        state[:, 5*i+4] = p

        if self.FLOAT_MODE == "bf16":
            wkv = (a / b).type(torch.bfloat16)
        elif self.FLOAT_MODE == "fp16":
            wkv = (a / b).half()
        else:
            wkv = a / b

        output = torch.empty_like(wkv)
        for batch_idx in range(batch_size):
            output[batch_idx] = mem[i][batch_idx](ow @ (r[batch_idx] * wkv[batch_idx]))

        return output
    
    def forward(self, ctx, state, mem1, mem2, preprocess_only=False):
        with torch.no_grad():
            w = self.w
            args = self.args

            batch_size = ctx.size(0)

            if self.args.vocab_size == 37:
                atan = ATan()
                x = atan(w.emb.weight[ctx[:, -1]])
            else:
                x = w.emb.weight[ctx[:, -1]]

            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()

            try:
                pos_emb = w.pos_emb[len(ctx[0])-1]  # Assuming ctx is a batch of sequences
                x = x + pos_emb
            except:
                pass

            if state is None:
                state = torch.zeros(batch_size, args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                mem1 = []
                mem2 = []

                for i in range(args.n_layer):
                    state[:, 5*i+4] -= 1e30
                    mem1.append([neuron.LIFNode() for _ in range(batch_size)])
                    mem2.append([neuron.LIFNode() for _ in range(batch_size)])

            for i in range(args.n_layer):
                if i == 0:
                    x = self.LN(x, w.blocks[i].ln0)

                ww = w.blocks[i].att
                att = self.SA(self.LN(x, w.blocks[i].ln1), state, i,
                              ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay,
                              ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight, mem1)
                x = x + att
                ww = w.blocks[i].ffn
                ffn = self.FF(self.LN(x, w.blocks[i].ln2), state, i,
                              ww.time_mix_k, ww.time_mix_r,
                              ww.key.weight, ww.value.weight, ww.receptance.weight, mem2)

                x = x + ffn
                if (i + 1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2

            if preprocess_only:
                return state, mem1, mem2

            x = self.LN(x, w.ln_out)
            x = torch.einsum('ij,bj->bi', w.head.weight, x)  # Batch matrix multiplication

            return x.float(), state, mem1, mem2