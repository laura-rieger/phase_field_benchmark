import itertools

import platform
from simple_slurm import Slurm
import subprocess

# check if we are on own computer or cluster
is_windows = platform.system() == "Windows"

#%%

batchSize = 64 if is_windows else 256
num_epochs = 1 if is_windows else 100
params_to_vary = {
    "seed": [x for x in range(1)],
    "num_epochs": [
        num_epochs,
    ],
    "prediction_offset": [
        5, 25, 125
    ],
        "num_input": [  # the weighing factor for the resampling
        3,

    ],
    "data_weight": [  # the weighing factor for the resampling
        50,

    ],
    "in_factor": [16,],
    "reduce_layer": [
        1,
    ],
    "exp_name": ["SweepNew3"],
    "start_offset": [
        50,
    ],
    "batch_size": [batchSize],
    "use_linear": [
        0,
    ],
}

keys = list(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples
print(len(param_combinations))
for i, _ in enumerate(param_combinations):

    curFunction = "python train_unet.py "

    for j, key in enumerate(keys):

        curFunction += "--" + key + " " + str(param_combinations[i][j]) + " "

    slurm = Slurm(
        mail_type="FAIL",
        partition="sm3090",
        N=1,
        n=8,
        time="1-01:00:00",
        gres="gpu:RTX3090:1",
    )
    if is_windows:
        subprocess.call(curFunction, shell=True)
        # print(cur_function)
    else:
        slurm.sbatch(curFunction)
