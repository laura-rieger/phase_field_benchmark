#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pathlib
import configparser

import numpy as np
from scipy import signal
from skimage.segmentation import flood_fill
from os.path import join as oj
from tqdm import tqdm

cur_path = pathlib.Path(__file__).parent.absolute()
os.chdir(cur_path)

config = configparser.ConfigParser()
config.read("../config.ini")


def test_trajectory(model, dataset, start, stop, offset, device="cpu", go_beyond=False):
    abs_err = np.zeros((dataset.get_num_traj()))
    outputs = np.zeros((dataset.get_num_traj(), 100, 100))
    output_list = []
    true_output_list = []
    true_outputs = np.zeros((dataset.get_num_traj(), 100, 100))
    with torch.no_grad():

        # num_steps = int((stop - start) / offset)
        for i in tqdm(range(dataset.get_num_traj())):
            # for i in range(10):
            cur_output_list = []
            cur_true_output_list = []

            first_img = dataset.get_trajectory(i, 0)

            cur_img = first_img[None, :].to(device)
            cur_output_list.append(cur_img.detach().cpu().numpy()[0, 0])
            cur_true_output_list.append(cur_img.detach().cpu().numpy()[0, 0])
            j=0
            while True:
                if dataset.get_len_traj(i) < j * offset and go_beyond == False:
                    break
                cur_img = cur_img.to(device)
          

                pred_out = model.forward(cur_img)

                cur_img = pred_out

                cur_output_list.append(cur_img.detach().cpu().numpy()[0, 0])
                # print(len(cur_output_list))
                try:
                    cur_true_output_list.append(
                        dataset.get_trajectory(i, (j + 1) * offset).detach().cpu().numpy()
                    )
                except IndexError:
                    pass
                j += 1

            output_list.append(cur_output_list)
            # print(len)
            true_output_list.append(cur_true_output_list)

            outputs[i] = cur_img.detach().cpu().numpy()
            true_outputs[i] = cur_true_output_list[-1] #dataset.get_trajectory(i, dataset.get_len_traj(i) - 1)

            output_large = np.tile(outputs[i], (2, 2))
            convolved = signal.correlate(output_large, true_outputs[i], mode="valid")
            my_indices = np.where(convolved == convolved.max())
            x_pos, y_pos = my_indices[0][0], my_indices[1][0]
            
            outputs[i] = output_large[x_pos : x_pos + 100, y_pos : y_pos + 100]

            abs_err[i] = np.abs(outputs[i] - true_outputs[i]).mean()
            # if i > 0:
            #     break
        # print(overlaps[i])
    return outputs, true_outputs, abs_err, output_list, true_output_list


def get_split(my_len, seed=42):
    split_idxs = np.arange(my_len)
    np.random.seed(seed)
    np.random.shuffle(split_idxs)

    (
        num_train,
        num_val,
    ) = (
        int(my_len * 0.7),
        int(my_len * 0.15),
    )
    num_test = my_len - num_train - num_val
    train_idxs = split_idxs[:num_train]
    test_idxs = split_idxs[num_train : num_train + num_val]
    val_idxs = split_idxs[-num_test:]
    return train_idxs, test_idxs, val_idxs


def get_info(file_name, folder_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    param_dict = {}
    for test_line in lines:
        if "=" in test_line:
            param_split = test_line.split("=")
            param_name = param_split[0].strip()
            for potential_param in param_split:
                try:
                    param_dict[param_name] = float(potential_param)
                except ValueError:
                    pass

        elif ":" in test_line:
            param_split = test_line.split(":")
            param_name = param_split[0]
            param_val = param_split[1].split(" ")[-1]
            param_val = float(param_val)
            param_dict[param_name] = param_val
    param_dict["foldername"] = folder_name
    return param_dict


def get_arr(file_name, width):
    with open(file_name, "r") as f:
        data_string = ""
        for line in f.readlines():
            data_string += line
        # string_list = [x for x in data_string.split(" ")[:-1]]
        # for my_string in string_list:
        #     try:
        #         float(my_string)
        #     except ValueError:
        #         print("stop")
        data_string = data_string.replace("\n", " ")

        my_strings = data_string.split(" ")
        my_strings = [x for x in my_strings if x != ""]
        data_num = [float(x) for x in my_strings]
        data_arr = np.asarray(data_num)
        assert len(data_arr) == width * width, "Data has size {}, expected {}".format(
            data_arr, width * width
        )
        return data_arr.reshape((width, width))


time_file_name = "Times_in_Seconds.txt"
result_file_name = "Result_Interpolated_Exp.txt"


def get_time_file(folder_name):
    with open(oj(folder_name, time_file_name), "r") as f:
        data_string = f.read()
        data_string = data_string.replace("\n", " ")
        data_num = [float(x) for x in data_string.split(" ") if x != ""]
        data_arr = np.asarray(data_num)
        return len(data_arr), data_arr


def read_file(folder_name):
    #     num_steps = get_time_file(folder_name)[0]
    with open(oj(folder_name, result_file_name), "r") as file:
        data_string = file.read()
    data_string = data_string.replace("\n", " ")
    my_strings = data_string.split(" ")
    my_strings = [x for x in my_strings if x != ""]
    data_num = [float(x) for x in my_strings]
    return np.asarray(data_num).reshape((-1, 50, 50))


def make_dataset(
    arr_list,
    begin_offset=20,
    prediction_offset=1,
    skip_step=1,
):
    f_it_x_list = []
    f_it_y_list = []
    for arr in arr_list:
        for i in range(begin_offset, len(arr) - prediction_offset, skip_step):
            if (
                np.maximum(arr[i].max(), arr[i + prediction_offset].max()) <= 1
                and np.minimum(arr[i].min(), arr[i + prediction_offset].min()) >= 0
            ):
                if np.abs(arr[i] - arr[i + prediction_offset]).sum() < 600:
                    f_it_x_list.append(arr[i])
                    f_it_y_list.append(arr[i + prediction_offset])

    return np.asarray(f_it_x_list), np.asarray(f_it_y_list)


def find_regions(img):
    uint_img = np.copy(img.astype(np.int16))
    found_pixels = np.zeros_like(uint_img, dtype=np.bool_)
    segmented_img = np.zeros_like(uint_img)
    while np.any(found_pixels == False):
        next_pixels = np.where(found_pixels == False)
        next_region = flood_fill(uint_img, (next_pixels[0][0], next_pixels[1][0]), 2)
        if uint_img[next_pixels[0][0], next_pixels[1][0]] == 1:
            segmented_img[np.where(next_region == 2)] = segmented_img.max() + 1
        else:
            segmented_img[np.where(next_region == 2)] = segmented_img.min() - 1
        found_pixels[np.where(next_region == 2)] = True

    identical_regions = np.where(
        (segmented_img[:, 0] * segmented_img[:, -1] > 0)
        * (segmented_img[:, 0] != segmented_img[:, -1])
    )[0]
    while len(identical_regions) > 0:
        first_identical_region = identical_regions[0]
        segmented_img[
            np.where(segmented_img == segmented_img[first_identical_region, -1])
        ] = segmented_img[first_identical_region, 0]
        identical_regions = np.where(
            (segmented_img[:, 0] * segmented_img[:, -1] > 0)
            * (segmented_img[:, 0] != segmented_img[:, -1])
        )[0]

    identical_regions = np.where(
        (segmented_img[0, :] * segmented_img[-1, :] > 0)
        * (segmented_img[0, :] != segmented_img[-1, :])
    )[0]
    while len(identical_regions) > 0:
        first_identical_region = identical_regions[0]
        segmented_img[
            np.where(
                segmented_img
                == segmented_img[
                    -1,
                    first_identical_region,
                ]
            )
        ] = segmented_img[
            0,
            first_identical_region,
        ]
        identical_regions = np.where(
            (segmented_img[0, :] * segmented_img[-1, :] > 0)
            * (segmented_img[0, :] != segmented_img[-1, :])
        )[0]
    num_pos_regions = segmented_img.max()
    avg_size = 0
    for i in range(1, num_pos_regions):
        avg_size += (segmented_img == i).sum()
    return segmented_img, num_pos_regions, avg_size / num_pos_regions


def measure_boundaries(img):
    horizontal = np.zeros_like(img, dtype=np.bool_)
    vertical = np.zeros_like(img, dtype=np.bool_)
    vertical[:, :-1] = (
        img[
            :,
            :-1,
        ]
        == img[
            :,
            1:,
        ]
    ) == False
    vertical[:, -1] = (
        img[
            :,
            -1,
        ]
        == img[
            :,
            1,
        ]
    ) == False

    horizontal[:-1] = (img[:-1,] == img[1:,]) == False
    horizontal[-1] = (img[-1,] == img[1,]) == False
    return (horizontal + vertical).sum(), (horizontal + vertical)


if __name__ == "__main__":
    pass
    # get_dataset(config['DATASET']['data_path'], 50)
