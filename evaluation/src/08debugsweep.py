import itertools

import platform
from simple_slurm import Slurm
import subprocess

# check if we are on own computer or cluster

is_windows = platform.system() == 'Windows'

if not is_windows:
    subprocess.run(["module load Python/3.11.3-GCCcore-12.3.0"], shell=True)     
    subprocess.run(["module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0"], shell=True) 
    subprocess.run(["module load virtualenv/20.23.1-GCCcore-12.3.0"], shell=True) 
    subprocess.run(["source ~/environments/basic_ml/bin/activate"], shell=True) 
#%%

batchSize = 64 if is_windows else 256

params_to_vary = {
    "tag": ["debugSweep"] ,
    "seed": [x for x in range(3)],
    "num_epochs": [ 200000000, ],
    "prediction_offset": [5,  ], # 10
    "num_input": [ 1, ],
    "data_weight": [-1,],
    "in_factor": [16,],
    "reduce_layer": [ 1, ],
    "exp_name": ["PhaseFieldPrediction"],
    "start_offset": [  20, ],
    "architecture" : [ "U-Net",],
    "loss": [ "mse",],
    "batch_size": [batchSize],
    "use_linear": [ 0, ],
    "train_percentage": [1., ],
}

keys = list(params_to_vary.keys())

vals = [params_to_vary[k] for k in keys]

param_combinations = list(itertools.product(*vals))  # list of tuples
print(len(param_combinations))
for i, _ in enumerate(param_combinations):

    curFunction = "python train.py "

    for j, key in enumerate(keys):

        curFunction += "--" + key + " " + str(param_combinations[i][j]) + " "

    slurm = Slurm(
        mail_type="FAIL",
        partition="sm3090el8",
        N=1,
        ntasks_per_node=8,
        time="0-03:00:00",
        gres="gpu:RTX3090:1",
    )
    if is_windows:
        subprocess.call(curFunction, shell=True)
        # print(cur_function)
    else:
        slurm.sbatch(curFunction)
