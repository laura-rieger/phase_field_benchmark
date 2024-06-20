import sys

import configparser
import pickle as pkl
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as oj
from unet_model import UNet
import torch
import os
from my_segformer import MySegformer
import torch.utils.data
from data_loading import H5pyDataset

# from torchvision import transforms
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
import data_utils
import platform

import numpy as np

print("started")


num_workers = 0 if platform.system() == "Windows" else 8

cuda = torch.cuda.is_available()
if not cuda:
    print("Cuda not available, exiting.")
    sys.exit(1)

device = torch.device("cuda" if cuda else "cpu")


def get_args():
    parser = ArgumentParser(description="U-Net Train")
    parser.add_argument("--exp_name", type=str, default="Debug")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=5000000)

    parser.add_argument("--train_percentage", type=float, default=1.0)
    parser.add_argument("--architecture", type=str, default="Segformer")

    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--num_input", type=int, default=1)
    parser.add_argument("--start_offset", type=int, default=10)
    parser.add_argument("--prediction_offset", type=int, default=50)

    parser.add_argument("--data_weight", type=int, default=1)
    parser.add_argument("--reduce_layer", type=int, default=1)
    parser.add_argument("--in_factor", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_linear", type=int, default=0)

    ret_args = parser.parse_args()
    return ret_args


args = get_args()


def test(
    my_model,
    optimizer,
    loader,
    loss_function,
):
    my_model.eval()
    tot_loss = 0

    with torch.no_grad():
        for in_imgs, out_imgs in loader:
            in_imgs = in_imgs.to(device)
            out_imgs = out_imgs.to(device)
            optimizer.zero_grad()
            pred_out = my_model.forward(in_imgs)[:, 0]
            loss = loss_function( pred_out, out_imgs, )
            optimizer.step()
            tot_loss += loss.item()

    return tot_loss / len(loader.dataset)


def train(my_model, optimizer, loader, loss_function):
    my_model.train()

    tot_loss = 0

    for in_imgs, out_imgs in loader:
        in_imgs = in_imgs.to(device)
        out_imgs = out_imgs.to(device)
        optimizer.zero_grad()
        pred_out = my_model.forward(in_imgs)[:, 0]
        loss = loss_function(pred_out, out_imgs)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss / len(loader.dataset)


def main():
    np.random.seed()
    results = {}
    file_name = "".join([str(np.random.choice(10)) for x in range(10)])


    results["file_name"] = file_name
    for arg in vars(args):
        if arg != "save_path":
            results[str(arg)] = getattr(args, arg)
    config = configparser.ConfigParser()


    if os.path.exists("../config.ini"):
        config.read("../config.ini")
    else:
        config.read("config.ini")
    data_folder = config["DATASET"]["dataPath"]
    model_folder = config["PATHS"]["model_unet_path"]

    train_dataset = H5pyDataset(
        oj(data_folder, "phase_field_train.h5"),
        offset=args.prediction_offset,
        start_offset=args.start_offset,
        scale=1,
        num_channels=args.num_input,
        percentage=args.train_percentage,
    )
    val_dataset = H5pyDataset(
        oj(data_folder, "phase_field_val.h5"),
        offset=args.prediction_offset,
        start_offset=args.start_offset,
        scale=1,
        num_channels=args.num_input,
    )
    if args.data_weight != -1:
        trainWeightList = np.concatenate( [
                (1 / np.arange(args.data_weight, num_data + args.data_weight))
                for num_data in train_dataset.num_list ] )
        valWeightList = np.concatenate(
            [
                (1 / np.arange(args.data_weight, num_data + args.data_weight))
                for num_data in val_dataset.num_list
            ]
        )

    else:
        trainWeightList = np.ones(len(train_dataset))
        valWeightList = np.ones(len(val_dataset))
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        trainWeightList, num_samples=60000
    )  # check training a bit more frequently
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        valWeightList, num_samples=20000
    )  
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        num_workers=num_workers,
        sampler=val_sampler,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("loaded data")

    if args.architecture.lower() == "segformer":
        my_model = MySegformer(num_input=args.num_input, num_classes=1, device=device)
    else:
        my_model = UNet(
            n_channels=args.num_input,
            n_classes=1,
            bilinear=True, # False
            in_factor=args.in_factor,
            use_small=(args.reduce_layer == 1),
        ).to(device=device)

    print("loaded model")
    if args.loss == "mse":
        loss_function = nn.MSELoss(reduction="sum")
    else:
        loss_function = nn.BCELoss(reduction="mean")

    optimizer = optim.AdamW( my_model.parameters(), )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=3
    )
    tr_losses = []
    val_losses = []

    best_val_loss = 510e15

    cur_patience = 0
    max_patience = 20  # max number of epochs to wait before stopping

    best_weights = None
    print("starting training")
    for _ in range(args.num_epochs):
        train_loss = train(my_model, optimizer, train_loader, loss_function)
        tr_losses.append(train_loss)
        val_loss = test(
            my_model,
            optimizer,
            val_loader,
            loss_function,
        )
        val_losses.append(val_loss)
        scheduler.step(val_loss)
  
        if val_losses[-1] < best_val_loss:
            best_weights = deepcopy(my_model.state_dict())
            cur_patience = 0

            best_val_loss = val_losses[-1]
            results["best_val_loss"] = best_val_loss
            results["train_losses"] = tr_losses
            results["val_losses"] = val_losses
            with open(oj(model_folder, file_name + ".pkl"), "wb") as f:
                pkl.dump(results, f)

            torch.save(best_weights, oj(model_folder, file_name + ".pt"))

        else:
            cur_patience += 1
        if cur_patience > max_patience:
            break

    my_model.load_state_dict(best_weights)

    results["prediction_err"] = data_utils.test_trajectory(
        my_model,
        val_dataset,
        args.start_offset,
        40000,
        args.prediction_offset,
        device=device,
    )[2].mean()

  

    with open(oj(model_folder, file_name + ".pkl"), "wb") as f:
        pkl.dump(results, f)

        torch.save(best_weights, oj(model_folder, file_name + ".pt"))

    test_dataset = H5pyDataset(
        oj(data_folder, "phase_field_test.h5"),
        offset=args.prediction_offset,
        start_offset=args.start_offset,
        scale=1,
        num_channels=args.num_input,
    )
    ood_dataset = H5pyDataset(
        oj(data_folder, "phase_field_ood.h5"),
        offset=args.prediction_offset,
        start_offset=args.start_offset,
        scale=1,
        num_channels=args.num_input,
    )
    out_test_err = data_utils.test_trajectory(
        my_model,
        test_dataset,
        args.start_offset,
        40000,
        args.prediction_offset,
        device=device,
    )[2]


    results["test_prediction_err"] = out_test_err.mean()

    nm_list = []
    for name in test_dataset.key_list:
        nm_list.append(int(name.split("_")[3][:-3]))




    results["ood_prediction_err"] = data_utils.test_trajectory(
        my_model,
        ood_dataset,
        args.start_offset,
        40000,
        args.prediction_offset,
        device=device,
    )[2].mean()


    with open(oj(model_folder, file_name + ".pkl"), "wb") as f:
        pkl.dump(results, f)
    split_dict = {}
    for nm_val, err_val in zip(nm_list, out_test_err):
        if nm_val not in split_dict and not np.isnan(err_val):
            split_dict[nm_val] = [err_val,]
        else:
            split_dict[nm_val].append(err_val)
    # probs not the best way but I am tired and coffe is not working
    split_dict = {key: np.mean(val) for key, val in split_dict.items()}
    results['split_accs'] = split_dict
    with open(oj(model_folder, file_name + ".pkl"), "wb") as f:
        pkl.dump(results, f)

    print("Saved")


if __name__ == "__main__":
    main()
