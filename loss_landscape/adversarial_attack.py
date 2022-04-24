import torch
import os
import dill

from copy import deepcopy
from load_data import get_dataloader
from utils.resnet import get_resnet
from utils.nn_manipulation import set_weights_from_swag_base
from swag.posteriors.swag import SWAG
from swag.utils import bn_update
from utils import swag_config


def save_adversarial_images(model_list, loader, save_folder, save_name, device, attack_type='pgd', eps=0.05,
                            alpha=0.05, iterations=20):
    os.makedirs(save_folder, exist_ok=True)
    model.to(device)
    model.eval()

    all_pred_labels = []
    labels_all = []
    for i, (images, labels) in enumerate(loader):
        print((i+1) * images.shape[0])
        images = images.to(device)
        labels = labels.to(device)

        images = attack_model(attack_type, model_list, images, labels, device, eps, alpha, iterations)

        pred_labels = model(images).argmax(axis=1).detach().cpu()
        all_pred_labels.append(pred_labels)
        labels_all.append(labels)

        save_path = os.path.join(save_folder, save_name + f'_{i}.pt')
        torch.save(dict(images=images.detach(), labels=labels.detach()), save_path)

    all_pred_labels = torch.cat(all_pred_labels)
    labels_all = torch.cat(labels_all)
    acc = (all_pred_labels.cpu() == labels_all.cpu()).float().mean().cpu().data.numpy()
    print(f"Accuracy of the model on the test images with {attack_type} attack: {100 * acc}%")


def attack_model(attack_type, model_list, images, labels, device, eps=0.05, alpha=0.05, iters=40):
    if attack_type == 'pgd':
        images = pgd_attack_on_best_model(model_list, images, labels, eps=eps, alpha=alpha, iterations=iters, device=device)
    else:
        assert True, 'Attack is unavailable, Using vanilla images'
    return images

def pgd_attack_on_best_model(model_list, images, labels, eps=0.1, alpha=2 / 255, iterations=40, device='cpu'):
    
    if not isinstance(model_list, list):
        model_list = [model_list]
    
    ori_images = images.data

    for _ in range(iterations):
        images.requires_grad = True
        costs = []
        for i, model in enumerate(model_list):
            outputs = model(images)
            model.zero_grad()
            costs.append(torch.nn.functional.cross_entropy(outputs.to(device), labels))

        cost = min(costs)
        # print(costs, cost)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = (ori_images + eta).detach_()

    return images

def pgd_attack_on_all_model(model_list, images, labels, eps=0.1, alpha=2 / 255, iterations=40, device='cpu'):
    
    if not isinstance(model_list, list):
        model_list = [model_list]
    
    ori_images = images.data

    for _ in range(iterations):
        images.requires_grad = True
        loss = 0
        for i, model in enumerate(model_list):
            outputs = model(images)
            model.zero_grad()
            loss += torch.nn.functional.cross_entropy(outputs.to(device), labels)

        # print(costs, cost)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = (ori_images + eta).detach_()

    return images

def save_swag_models(model, swag_model, loader, swag_num_samples, swag_scale, use_swag_diag_cov, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    model_list = []
    for model_id in range(swag_num_samples):
        swag_model.sample(scale=swag_scale, cov=not use_swag_diag_cov)
        bn_update(loader, swag_model, verbose=True, subset=0.1)
        set_weights_from_swag_base(model, swag_model.base)
        
        path = os.path.join(save_dir, f'{model_id+1}_model.pt')
        torch.save(model.state_dict(), path, pickle_module=dill)
        
        model_list.append(deepcopy(model))
    return model_list

def load_swag_models(path, model, device):
    model_list = []
    for model_file in os.listdir(path):
        model.load_state_dict(torch.load(os.path.join(path, model_file), map_location=device))
        model_list.append(deepcopy(model))
    return model_list

if __name__ == '__main__':
    
    model_folder = r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag' 
    model_dir = r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag\ckpt'
    
    save_swag_samples_folder = os.path.join(model_folder, 'swag_samples')
    save_adversarial_folder = os.path.join(model_folder, 'adversarial_data_best_model')
    train = True
    
    ckpt_load = 200
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_name = 'resnet20'
    num_classes = 10
    remove_skip_connections = False
    
    use_swag_models = True
    
    train_loader, test_loader = get_dataloader(batch_size=1)
    
    if train:
        save_folder = os.path.join(save_adversarial_folder, 'train') 
        loader = train_loader
    else:
        save_folder = os.path.join(save_adversarial_folder, 'test') 
        loader = test_loader
    os.makedirs(save_folder, exist_ok=True)
    
    model = get_resnet(model_name)(num_classes=num_classes, remove_skip_connections=remove_skip_connections)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{ckpt_load}_model.pt"), map_location=device))
    
    if use_swag_models:
        print("Using Bayesian Models")
        swag_samples_folder = os.path.join(model_folder, 'swag_samples')
        if os.path.exists(swag_samples_folder) and len(os.listdir(swag_samples_folder)) == swag_config.NUM_SAMPLES:
            print("SWAG models exist [load models]")
            model_list = load_swag_models(swag_samples_folder, model, device)
        else:
            print("SWAG models don't exist or not enough [create and save models]")
            # load swag model
            swag_model_file = os.path.join(model_folder, 'swag_ckpt', f'{ckpt_load}_swag_model.pt')
            swag_model = SWAG(get_resnet(model_name),
                            num_classes=10, remove_skip_connections=remove_skip_connections, 
                            no_cov_mat=swag_config.USE_DIAG_COV, max_num_models=ckpt_load, var_clamp=swag_config.VAR_CLIP)
            state_dict = torch.load(swag_model_file, pickle_module=dill, map_location=device)
            swag_model.load_state_dict(state_dict)
            swag_model.to(device)
            # update batchnorm statistics
            swag_model.sample(0.0)
            bn_update(loader, swag_model, verbose=True, subset=0.1)

            # get and save swag samples
            model_list = save_swag_models(model=model, 
                                        swag_model=swag_model,
                                        loader=loader, 
                                        swag_num_samples=swag_config.NUM_SAMPLES, 
                                        swag_scale=swag_config.SCALE, 
                                        use_swag_diag_cov=swag_config.USE_DIAG_COV, 
                                        save_dir=save_swag_samples_folder)
    else:
        print("Using Saved Model")
        model_list = [model]
    print(f'Number of models = {len(model_list)}')
    
    # print(get_loss_value(model_list, loader, device, attack_type=None, eps=0.05, alpha=0.05, iterations=20))
    save_adversarial_images(model_list=model_list, loader=loader, save_folder=save_adversarial_folder, 
                            save_name='adversarial_image', device=device)
   