import argparse
import logging
import os
import pprint
from re import L
import time
import json
import numpy as np
import dill
import numpy.random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

# from utils.evaluations import get_loss_value
from utils.linear_algebra import FrequentDirectionAccountant
from utils.nn_manipulation import count_params, flatten_grads
from utils.reproducibility import set_seed
from utils.resnet import get_resnet, set_resnet_weights
import matplotlib.pyplot as plt

from adversarial_attack import pgd_attack_l2, pgd_attack
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_adversarial_images(file_dir):
    data = torch.load(file_dir)
    return data["images"], data["labels"]

def test_model():
    root_dir = 'C:/Users/berkt/Desktop/cs699_dynamics_of_representation_learning/loss_landscape/results_final/'
    model_name = 'resnet20'
    remove_skip_connections = False
    model_dir = root_dir + 'resnet20_skip_bn_bias'
    ckpt_load = 200
    device = 'cuda:0'
    seed = 0
    batch_size = 1

    data_dir = root_dir + "resnet20_skip_bn_bias/"
    save_name = 'pgd_image_'
    train = True

    if train:
        number_of_files = 50
        data_type_str = 'train'
    else:
        number_of_files = 10
        data_type_str = 'test'
        
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    model = get_resnet(model_name)(
            num_classes=10, remove_skip_connections=remove_skip_connections
        )

    # load saved model
    state_dict = torch.load(f"{model_dir}/ckpt/{ckpt_load}_model.pt", map_location=device)
    model = set_resnet_weights(model, state_dict)
    model.to(device)

    model.eval()

    all_pred_labels = []
    all_labels = []

    for i in range(number_of_files):
        images, labels = load_adversarial_images(data_dir + f'adv_{data_type_str}_data_pgd' + f"/pgd_image_{i}.pt")
        
        images = images.to(device)
        labels = labels.to(device)
        
        pred_labels = model(images).argmax(axis=1).detach().to('cpu')

        all_pred_labels.append(pred_labels)
        all_labels.append(labels.detach())

    all_pred_labels = torch.cat(all_pred_labels)
    all_labels =  torch.cat(all_labels)

    acc = (all_labels.cpu() == all_pred_labels.cpu()).float().mean().data.numpy()

    print(f"Test Accuracy on \n\
            vanilla images: {acc}")


if __name__ == '__main__':
    test_model()