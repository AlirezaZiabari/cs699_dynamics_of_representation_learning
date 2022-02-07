"""
Script to attack a neural network:
"""

import torch

# miss classify the labels
def pgd_attack(model, images, labels, eps=0.1, alpha=2/255, iters=40) :        
    ori_images = images.data
        
    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = torch.nn.functional.cross_entropy(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

def norm_fn(x):
    return x.view(x.shape[0], -1).norm(dim=1)[:,None,None,None]


def pgd_attack_l2(model, images, labels, eps=0.1, alpha=2/255, iters=40, delta_init_type='zeros'):
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
      loss = torch.nn.functional.cross_entropy(model(images + delta), labels)
      loss.backward()

      delta.data += alpha*delta.grad.detach() / norm_fn(delta.grad.detach())
      delta.data = torch.min(torch.max(delta.detach(), - images), 1 - images) # clip X+delta to [0,1]
      delta.data *= eps / norm_fn(delta.detach()).clamp(min=eps)
      delta.grad.zero_()
  
  delta_final = delta.detach()
  pert_images = delta_final + images
  return pert_images

