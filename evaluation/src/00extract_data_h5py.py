import os, sys

sys.path.insert(0, "src")
sys.path.insert(0, "../src")

from os.path import join as oj
from tqdm import tqdm
import numpy as np
import configparser
import h5py

config = configparser.ConfigParser()
# config is either in this folder or the parent
if os.path.isfile('../config.ini'):
    config.read('../config.ini')
else:
    config.read('./config.ini')
dataPath = config['DATASET']['dataPath']
jobPath = config['DATASET']['jobPath']

h5_file_name = 'phase_field.h5'
# h5_file_name = 'phase_field_ood.h5'
full_file_name = oj(dataPath, h5_file_name)


def reformat(x):
    dataString = x.replace("\n", " ")
    dataStrings = dataString.split(" ")
    dataStrings = [x for x in dataStrings if x != '']
    dataNum = [ float(x.replace('*^-', 'e-').replace('*^', '')) for x in dataStrings ]
    concentrationImage = np.reshape(np.asarray(dataNum), (10000, 3))
    concentrationImage = concentrationImage[:, 2].reshape(100, 100)
    return concentrationImage


jobFolders = [x for x in os.listdir(jobPath) if "FAILED" not in x]
skip_job = False



for folder in jobFolders:
    print(folder)
    with h5py.File(full_file_name, 'a') as h5file:
        curJobFolder = oj(jobPath, folder)
        already_in_h5 = list(h5file.keys())
        case_names = os.listdir(curJobFolder)

        for j,caseName in tqdm(enumerate(case_names), position=0, leave=True):
            if caseName in already_in_h5:
                # print(caseName)
                continue
            try:

                animationFolder = oj(curJobFolder, caseName, 'Animation')
                listOfFiles = [ x for x in os.listdir(animationFolder) if 'concentration_' in x ]
                numFiles = len(listOfFiles)
     
                new_arr = np.zeros((numFiles, 100, 100))

                for i in tqdm(range(numFiles), position=1,  leave=False):

                    with open(oj(animationFolder, 'concentration_' + str(i) + '.txt'),
                            'r') as f:

                        curArray = reformat(f.read())
                        new_arr[i] = curArray
                print("Done")

                h5file.create_dataset(caseName, data=new_arr)
            except:
                print(caseName)
                continue
