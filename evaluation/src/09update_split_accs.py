#This script is not needed anymore but originally the accuracy was not calculated for each domain size. 
#This script was used to calculate the accuracy for each domain size and add it to the results file


import itertools

import platform
from simple_slurm import Slurm
import subprocess

# check if we are on own computer or cluster

is_windows = platform.system() == 'Windows'

if not is_windows:
    subprocess.run(["module load Python/3.11.3-GCCcore-12.3.0"], shell=True) 
    subprocess.run(["source ~/environments/basic_ml/bin/activate"], shell=True) 


curFunction = "python calculate_split_accuracies.py"


slurm = Slurm(
    mail_type="FAIL",
    partition="sm3090el8",
    N=1,
    ntasks_per_node=8,
    time="0-10:00:00",
    gres="gpu:RTX3090:1",
)
if is_windows:
    subprocess.call(curFunction, shell=True)
    # print(cur_function)
else:
    slurm.sbatch(curFunction)
