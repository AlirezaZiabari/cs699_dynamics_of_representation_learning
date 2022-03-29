import torch
import os

from load_data import get_dataloader
from utils.resnet import get_resnet


def save_adversarial_images(model, loader, save_folder, save_name, device, attack_type='pgd', eps=0.05,
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

        images = attack_model(attack_type, model, images, labels, device, eps, alpha, iterations)

        pred_labels = model(images).argmax(axis=1).detach().cpu()
        all_pred_labels.append(pred_labels)
        labels_all.append(labels)

        save_path = os.path.join(save_folder, save_name + f'_{i}.pt')
        torch.save(dict(images=images.detach(), labels=labels.detach()), save_path)

    all_pred_labels = torch.cat(all_pred_labels)
    labels_all = torch.cat(labels_all)
    acc = (all_pred_labels.cpu() == labels_all.cpu()).float().mean().cpu().data.numpy()
    print(f"Accuracy of the model on the test images with {attack_type} attack: {100 * acc}%")


def attack_model(attack_type, model, images, labels, device, eps=0.05, alpha=0.05, iters=40):
    if attack_type == 'pgd_l2':
        images = pgd_l2_attack(model, images, labels, eps=eps, alpha=alpha, iterations=iters, device=device)
    elif attack_type == 'pgd':
        images = pgd_attack(model, images, labels, eps=eps, alpha=alpha, iterations=iters, device=device)
    else:
        assert True, 'Attack is unavailable, Using vanilla images'
    return images


def pgd_attack(model, images, labels, eps=0.1, alpha=2 / 255, iterations=40, device='cpu'):
    ori_images = images.data

    for i in range(iterations):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()

        cost = torch.nn.functional.cross_entropy(outputs.to(device), labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = (ori_images + eta).detach_()

    return images


def pgd_l2_attack(model, images, labels, eps=0.1, alpha=2 / 255, iterations=40, delta_init_type='zeros', device='cpu'):
    """
    Generates perturbed images for adversarial attack, with pgd_l2 (ref:https://adversarial-ml-tutorial.org/adversarial_examples/)
    Inputs:
    x: Input images (num_images, width, height, 3)
    y: Output labels (num_images, )
    eps: Value to clip delta (small)
    alpha: Adversarial budget, gradient ascent learning rate
    num_steps: Number of steps to iterate gradient ascent, should be comparable with eps/alpha
    delta_init_type: Initialization type of delta, either 'zeros' or 'random'

    Outputs:
    pert_images: perturbed images
    """
    def norm_fn(x):
        return x.view(x.shape[0], -1).norm(dim=1)[:, None, None, None]

    if delta_init_type == 'zeros':
        delta = torch.zeros_like(images, requires_grad=True)
    elif delta_init_type == 'random':
        delta = torch.rand_like(images, requires_grad=True)

    for i in range(iterations):
        loss = torch.nn.functional.cross_entropy(model(images + delta).to(device), labels)
        loss.backward()

        delta.data += alpha * delta.grad.detach() / norm_fn(delta.grad.detach())

        delta.data *= eps / norm_fn(delta.detach()).clamp(min=eps)
        delta.grad.zero_()

    delta_final = delta.detach()
    pert_images = delta_final + images
    return pert_images



if __name__ == '__main__':
    
    save_folder = r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag\adversarial_data' 
    model_dir = r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag\ckpt'
    train = True

    ckpt_load = 200
    device = 'cuda:0'
    train_loader, test_loader = get_dataloader(batch_size=1)
    
    if train:
        save_folder = os.path.join(save_folder, 'train') 
        loader = train_loader
    else:
        save_folder = os.path.join(save_folder, 'test') 
        loader = test_loader
    os.makedirs(save_folder, exist_ok=True)
    
    
    model = get_resnet('resnet20')(num_classes=10, remove_skip_connections=False)
    # model = torch.nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{ckpt_load}_model.pt"), map_location=device))
    
    save_adversarial_images(model=model, loader=loader, save_folder=save_folder, save_name='adversarial_image', device=device)
    
    
    
    