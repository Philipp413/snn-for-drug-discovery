import torch
import lava_dl.src.lava.lib.dl.slayer as slayer
import lava_dl.src.lava.lib.dl.bootstrap as bootstrap
import matplotlib as plt
import h5py

class SLAYERDense(torch.nn.Module):
    def __init__(self, trial,input_size):
        super(SLAYERDense, self).__init__()
        neuron_params = {
                'threshold'     : trial.suggest_float("threshold",0,2),
                'current_decay' : trial.suggest_float("current_decay",0,1),
                'voltage_decay' : trial.suggest_float("voltage_decay",0.6,1),
                'tau_grad'      : trial.suggest_float("tau_grad",0,1),
                'scale_grad'    : trial.suggest_float("scale_grad",0.5,4),
                'requires_grad' : trial.suggest_categorical('require_grad', [True,False]),     
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05),}
        num_input_neurons = input_size
        weight_norm = trial.suggest_categorical('Weight_norm', [True,False])
        delay = trial.suggest_categorical('Delay', [True,False])
        num_hidden = trial.suggest_int("num_hidden",512,2048,step=128)
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(neuron_params_drop, num_input_neurons, num_hidden, weight_norm=weight_norm, delay=delay), 
                slayer.block.cuba.Dense(neuron_params_drop, num_hidden, num_hidden, weight_norm=weight_norm, delay=delay), 
                slayer.block.cuba.Dense(neuron_params, num_hidden, 2, weight_norm=weight_norm), 
            ])
    
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike
    
    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
    
    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

class ANNDense(torch.nn.Module):
    def __init__(self, trial,input_size):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        num_hidden = trial.suggest_int("num_hidden",512,2048,step=128)
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_hidden), # 1024 is the number of input features
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits