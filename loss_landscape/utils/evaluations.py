import logging
import os
import sys
import dill
import numpy
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'loss_landscape'))

from sklearn.decomposition import PCA
from torch import nn

from utils.nn_manipulation import count_params, flatten_params
from adversarial_attack import attack_model
from load_data import load_data_from_path
from utils.resnet import set_resnet_weights

logger = logging.getLogger()


def get_loss_value(model, loader, device, attack_type, eps, alpha, iterations):
    """
    Evaluation loop for the multi-class classification problem.

    return (loss, accuracy)
    """

    model.eval()
    losses = []
    accuracies = []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if attack_type is not None:
            images = attack_model(attack_type, model, images, labels, device,
                                  eps, alpha, iterations)
        # Forward pass, vanilla
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels, reduce=None).detach()
        losses.append(loss.reshape(-1))

        acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
        accuracies.append(acc.reshape(-1))

    losses = torch.cat(losses, dim=0).mean().cpu().data.numpy()
    accuracies = torch.cat(accuracies, dim=0).mean().cpu().data.numpy()

    return losses, accuracies


def get_loss_value_for_saved_data(model, path, device):
    """
    Evaluation loop for the multi-class classification problem.

    return (loss, accuracy)
    """

    model.eval()
    losses = []
    accuracies = []
    list_dir = os.listdir(path)
    for batch_dir in list_dir:
        images, labels = load_data_from_path(path + '/' + batch_dir)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass, vanilla
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels, reduce=None).detach()
        losses.append(loss.reshape(-1))

        acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
        accuracies.append(acc.reshape(-1))

    losses = torch.cat(losses, dim=0).mean().cpu().data.numpy()
    accuracies = torch.cat(accuracies, dim=0).mean().cpu().data.numpy()

    return losses, accuracies


def get_PCA_directions(model: nn.Module, state_files, skip_bn_bias):
    """
        Compute PCA direction as defined in Li et al. 2017 (https://arxiv.org/abs/1712.09913)
    :param model: model object
    :param state_files: list of checkpoints.
    :param skip_bn_bias: Skip batch norm and bias while flattening the model params. Li et al. do not use batch norm and bias parameters
    :return: (pc1, pc2, explained variance)
    """

    # load final weights and flatten
    state_dict = torch.load(state_files[-1], pickle_module=dill, map_location="cpu")
    model = set_resnet_weights(model, state_dict)
    total_param = count_params(model, skip_bn_bias=skip_bn_bias)
    w_final = flatten_params(model, total_param, skip_bn_bias=skip_bn_bias)

    # compute w_i- w_final
    w_diff_matrix = numpy.zeros((len(state_files) - 1, total_param))
    for idx, file in enumerate(state_files[1:-1]):
        state_dict = torch.load(file, pickle_module=dill, map_location="cpu")
        model = set_resnet_weights(model, state_dict)
        w = flatten_params(model, total_param, skip_bn_bias=skip_bn_bias)

        diff = w - w_final
        w_diff_matrix[idx] = diff

    # Perform PCA on the optimization path matrix
    logger.info("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(w_diff_matrix)
    pc1 = numpy.array(pca.components_[0])
    pc2 = numpy.array(pca.components_[1])
    logger.info(
        f"angle between pc1 and pc2: {numpy.dot(pc1, pc2) / (numpy.linalg.norm(pc1) * numpy.linalg.norm(pc2))}"
    )
    logger.info(f"pca.explained_variance_ratio_: {pca.explained_variance_ratio_}")

    return pc1, pc2, pca.explained_variance_ratio_
