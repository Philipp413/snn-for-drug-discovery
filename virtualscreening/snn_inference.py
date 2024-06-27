import numpy as np
import os, json
import logging
import matplotlib.pyplot as plt
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.utils.system import Loihi2
from lava.lib.dl import netx
from src.datasets import *
import importlib

os.environ["SLURM"] = "1"
os.environ["LOIHI_GEN"] = "N3B3" # "N3C1"
os.environ["PARTITION"] = "nahuku08"

Loihi2.preferred_partition = 'nahuku08'
loihi2_is_available = Loihi2.is_loihi2_available

if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.utils import loihi2_profiler
else:
    RuntimeError("Loihi2 compiler is not available in this system. "
                 "This tutorial cannot proceed further.")

study_name = f"SLAYER_dense_morgan1024xT"
results_path=f'./experiments/{study_name}/'
os.makedirs(results_path, exist_ok=True)
model_path = f"./model/{study_name}.net"

net = netx.hdf5.Network(net_config=model_path ,
                        reset_interval=16,
                        reset_offset=0)

print(net)

steps_per_sample = net.reset_interval
num_steps = 100000  # Run for a very long time to get good power measurement

dataset_path = "/homes/pkueppers/data/MoleculeACE/CHEMBL234_Ki.csv"

module_name = importlib.import_module("src.encodings")
encoding = "morgan1024xT"
function_encoding = getattr(module_name,encoding)

full_set = SMILESDataset(path=dataset_path,mode="test",transform=function_encoding)

frame_id = 9
smile, gt = full_set[frame_id]
net.in_layer.neuron.bias = smile

power_logger = loihi2_profiler.Loihi2Power(num_steps=num_steps)
runtime_logger = loihi2_profiler.Loihi2ExecutionTime()

# also run 100 times and take average?

run_config = Loihi2HwCfg(callback_fxs=[power_logger, runtime_logger])
net._log_config.level = logging.INFO
net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
net.stop()

# measuring execution time
# 1e6 or 1e-6 is used to convert from microseconds to seconds
inference_rate = 1e6 / runtime_logger.avg_time_per_step / steps_per_sample
time_per_sample = (runtime_logger.avg_time_per_step * steps_per_sample) / 1e6
total_inference_time = num_steps * runtime_logger.avg_time_per_step * 1e-6

print(f'Average time per sample: {time_per_sample:.2f} s')
print(f'Throughput : {inference_rate:.2f} predictions per second.')
print(f'Total inference time: {total_inference_time:.2f} s')

# power measurements
vdd_p = power_logger.vdd_power  # neurocore power
vddm_p = power_logger.vddm_power  # memory power
vddio_p = power_logger.vddio_power  # IO power
total_power = power_logger.total_power

num_chips = 1 # according to https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf, oheogulch has only one intel loihi 2 chip
if Loihi2.partition in ['kp', 'kp_stack', 'kp_build', 'nahuku08']:
    num_chips = 8
if Loihi2.partition in ["nahuku32"]:
    num_chips = 32

# so I execute everything on only one loihi processor

# per chip static power
static_total_power = power_logger.static_total_power / num_chips
static_vdd_p = power_logger.static_vdd_power / num_chips
static_vddm_p = power_logger.static_vddm_power / num_chips
static_vddio_p = power_logger.static_vddio_power / num_chips

# compensate for static power of multiple chip
total_power -= (num_chips - 1) * static_total_power
vdd_p -= (num_chips - 1) * static_vdd_p
vddm_p -= (num_chips - 1) * static_vddm_p
vddio_p -= (num_chips - 1) * static_vddio_p

total_power_mean = np.mean(total_power)
vdd_p_mean = np.mean(vdd_p)
vddm_p_mean = np.mean(vddm_p)
vddio_p_mean = np.mean(vddio_p)
print(f'Total Power   : {total_power_mean:.6f} W')
print(f'Dynamic Power : {total_power_mean - static_total_power:.6f} W')
print(f'Static Power  : {static_total_power:.6f} W')
print(f'VDD Power     : {vdd_p_mean:.6f} W')
print(f'VDD-M Power   : {vddm_p_mean:.6f} W')
print(f'VDD-IO Power  : {vddio_p_mean:.6f} W')

total_energy = total_power_mean / inference_rate
dynamic_energy = (total_power_mean - static_total_power) / inference_rate
print(f'Total Energy per inference   : {total_energy * 1e3:.6f} mJ')
print(f'Dynamic Energy per inference : {dynamic_energy * 1e3:.6f} mJ')

results = {"time_per_sample": time_per_sample, "predictions_per_second": inference_rate, "dynamic_enery_per_sample" : dynamic_energy, "total_energy_per_sample": total_energy}
with open(f'{results_path}{study_name}_{frame_id}.json', "w") as f:
    print("saving results")
    json.dump(results, f, indent=4)