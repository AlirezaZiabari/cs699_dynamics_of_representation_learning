from email.quoprimime import unquote
import torch
import os 
import numpy as np

def load_adversarial_images(load_folder, load_name, sample_size=10000):
    labels = []
    for i in range(sample_size):
        load_path = os.path.join(load_folder, load_name + f'_{i}.npy')
        data = np.load(load_path, allow_pickle=True)
        labels.append(data.item()['labels'])
    
    unique_labels, count = np.unique(labels, return_counts=True)
    print(f'Labels: {unique_labels}, count: {count}')
    

def save_adversarial_images(model, loader, save_folder, save_name, device, attack_type='pgd', attack_eps=0.05, attack_alpha=0.05, attack_iters=20):
    
    os.makedirs(save_folder, exist_ok=True)
    model.to(device)
    
    pred_labels_all = []
    labels_all = []
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        if attack_type == 'pgd_l2':
            images = pgd_attack_l2(model, images, labels, eps=attack_eps, alpha=attack_alpha, iters=attack_iters, device=device)
        elif attack_type == 'pgd':
            images = pgd_attack(model, images, labels, eps=attack_eps, alpha=attack_alpha, iters=attack_iters, device=device)
        
        pred_labels = model(images).argmax(axis=1).detach().to('cpu')
        pred_labels_all.append(pred_labels)
        labels_all.append(labels)
        
        save_path = os.path.join(save_folder, save_name + f'_{i}.pt')
        torch.save(dict(images=images.detach(), labels=labels.detach()), save_path)
        # np.save(save_path, dict(images=images.detach().to('cpu'), labels=labels.detach().to('cpu')), allow_pickle=True)
    pred_labels_all = torch.cat(pred_labels_all)
    labels_all = torch.cat(labels_all)
    acc = (pred_labels_all.cpu() == labels_all.cpu()).float().mean().cpu().data.numpy()
    print(acc)

def pgd_attack(model, images, labels, eps=0.1, alpha=2/255, iters=40, device='cpu') :            

    ori_images = images.data
        
    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()

        cost = torch.nn.functional.cross_entropy(outputs.to(device), labels)
        cost.backward()
        
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images = (ori_images + eta).detach_()
              
    return images


def norm_fn(x):
    return x.view(x.shape[0], -1).norm(dim=1)[:,None,None,None]



def pgd_attack_l2(model, images, labels, eps=0.1, alpha=2/255, iters=40, delta_init_type='zeros', device='cpu'):

  '''
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
  '''

  if delta_init_type == 'zeros':
      delta = torch.zeros_like(images, requires_grad=True)
  elif delta_init_type == 'random':
      delta = torch.rand_like(images, requires_grad=True)

  for i in range(iters):
      loss = torch.nn.functional.cross_entropy(model(images + delta).to(device), labels)
      loss.backward()

      delta.data += alpha*delta.grad.detach() / norm_fn(delta.grad.detach())

    #   delta.data = torch.min(torch.max(delta.detach(), - images), 1 - images) # clip X+delta to [0,1]

      delta.data *= eps / norm_fn(delta.detach()).clamp(min=eps)
      delta.grad.zero_()
  
  delta_final = delta.detach()
  pert_images = delta_final + images
  return pert_images

