import os, sys
sys.path.insert(0, "src")
sys.path.insert(0, "../src")
from os.path import join as oj
import numpy as np
import configparser
import h5py


config = configparser.ConfigParser()
# config is either in this folder or the parent
if os.path.isfile('../config.ini'):
    config.read('../config.ini')
else:
    config.read('./config.ini')



# Set a fixed random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
dataPath = config['DATASET']['dataPath']


# Define the paths for input and output HDF5 files
input_h5_file = oj(dataPath, 'phase_field.h5')
output_train_h5_file = oj(dataPath, 'phase_field_train.h5')
output_val_h5_file = oj(dataPath, 'phase_field_val.h5')
output_test_h5_file = oj(dataPath, 'phase_field_test.h5')

# Open the input HDF5 file for reading
with h5py.File(input_h5_file, 'r') as input_file:
    # Create or open the output HDF5 files for writing
    with h5py.File(output_train_h5_file, 'w') as train_file, \
         h5py.File(output_val_h5_file, 'w') as val_file, \
         h5py.File(output_test_h5_file, 'w') as test_file:
                # Get a list of dataset names in the input HDF5 file
        dataset_names = sorted(list(input_file.keys()))
        
        # Shuffle the dataset names to randomly allocate them
        np.random.shuffle(dataset_names)
        num_datasets = len(dataset_names)
        num_train, num_val = int(0.7 * num_datasets), int(0.15 * num_datasets)
        num_test = num_datasets - num_train - num_val
        
        train_datasets = dataset_names[:num_train]
        val_datasets = dataset_names[num_train:num_train+num_val]
        test_datasets = dataset_names[-num_test:]
        
        
        # Copy datasets to the output files based on allocation
        for dataset_name in train_datasets:
            train_file.copy(input_file[dataset_name], dataset_name)
        for dataset_name in val_datasets:
            val_file.copy(input_file[dataset_name], dataset_name)
        for dataset_name in test_datasets:
            test_file.copy(input_file[dataset_name], dataset_name)

print("Data allocation and saving completed.")