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

def get_loss_value(model_list, loader, device, attack_type=None, eps=0.05, alpha=0.05, iterations=20):
    
    """
    Evaluation loop for the multi-class classification problem.

    return (loss, accuracy)
    """
    if not isinstance(model_list, list):
        model_list = [model_list]
    
    for model in model_list:
        model.eval()
    
    num_models = len(model_list)
    
    losses = []
    accuracies = []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if attack_type is not None:
            # TODO: which model should we do the attack on?
            images = attack_model(attack_type, model, images, labels, device,
                                eps, alpha, iterations)
            
        # Forward pass, vanilla
        outputs = None
        for model in model_list:
            model = model.to(device)
            if outputs is None:
                outputs = torch.nn.functional.softmax(model(images), dim=1) / num_models
            else:
                outputs += torch.nn.functional.softmax(model(images), dim=1) / num_models
        
        log_outputs = torch.nan_to_num(torch.log(outputs), neginf=-500)
        
        loss =  torch.nn.functional.nll_loss(log_outputs, labels, reduce=None).detach()

        losses.append(loss.reshape(-1))

        acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
        accuracies.append(acc.reshape(-1))

    loss = torch.cat(losses, dim=0).mean().cpu().data.numpy()
    accuracy = torch.cat(accuracies, dim=0).mean().cpu().data.numpy()
    
    return loss, accuracy



def get_loss_value_for_saved_data(model_list, path, device):
    """
    Evaluation loop for the multi-class classification problem.

    return (loss, accuracy)
    """

    for model in model_list:
        model.eval()
    
    num_models = len(model_list)
    
    losses = []
    accuracies = []
    list_dir = os.listdir(path)
    for batch_dir in list_dir:
        images, labels = load_data_from_path(path + '/' + batch_dir)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass, vanilla
        outputs = None
        for model in model_list:
            if outputs is None:
                outputs = torch.nn.functional.softmax(model(images), dim=1) / num_models
            else:
                outputs += torch.nn.functional.softmax(model(images), dim=1) / num_models
        
        loss = torch.nn.functional.cross_entropy(outputs, labels, reduce=None).detach()
        losses.append(loss.reshape(-1))

        acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
        accuracies.append(acc.reshape(-1))

    loss = torch.cat(losses, dim=0).mean().cpu().data.numpy()
    accuracy = torch.cat(accuracies, dim=0).mean().cpu().data.numpy()
    
    return loss, accuracy


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
