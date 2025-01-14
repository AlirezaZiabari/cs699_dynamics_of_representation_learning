import torch
from collections import OrderedDict

def set_weights_from_swag_base(model, swag_base):
    
    state_dict = OrderedDict()
    
    # first load non-BN related parameters except BN weight and bias
    for i, (name, param) in enumerate(model.named_parameters()):
        split_name = name.split('.')
        if 'layer' not in name:
            if split_name[-1] == 'weight':
                state_dict[name] = swag_base._modules[split_name[0]].weight
            else:
                state_dict[name] = swag_base._modules[split_name[0]].bias
        else:
            if split_name[-1] == 'weight':
                state_dict[name] = swag_base._modules[split_name[0]][int(split_name[1])]._modules[split_name[2]].weight
            else:
                state_dict[name] = swag_base._modules[split_name[0]][int(split_name[1])]._modules[split_name[2]].bias
    
    # Load BN running mean and variance parameters
    for (name, param) in swag_base.state_dict().items():    
        if 'running' in name or 'num_batches' in name:
            state_dict[name] = param
    
    model.load_state_dict(state_dict)
            
        
def set_weights_by_direction(model, x, y, direction1, direction2, weights, skip_bn_bias=False):
    if direction2 is not None:
        changes = direction1 * x + direction2 * y
    else:
        changes = direction1 * x

    apply_params(model, weights + changes, skip_bn_bias=skip_bn_bias)


def create_normalized_random_direction(model, skip_bn_bias=False):
    weights = [param.data for param in model.parameters()]
    directions = []

    # filter normalization part
    for w in weights:
        if w.dim() <= 1:
            if skip_bn_bias:
                pass
                # ignore directions for weights with 1 dimension
            else:
                # this is different from original paper. We just keep sane defaults here
                t = torch.randn_like(w)
                t.mul_(torch.abs(t) / (torch.abs(w)) + 1e-10)
                directions.append(t)
        else:
            d = torch.randn_like(w)
            for filter_d, filter_w in zip(d, w):
                filter_d.mul_(filter_w.norm() / (filter_d.norm() + 1e-10))
            directions.append(d)
    return directions


def count_params(model: torch.nn.Module, skip_bn_bias=False):
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() <= 1 and skip_bn_bias:
                pass
            else:
                count += param.numel()
    return count


def flatten_params(model, num_params, skip_bn_bias=False):
    flat_param = torch.zeros(num_params, requires_grad=False)
    idx = 0
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() <= 1 and skip_bn_bias:
                pass
            else:
                flat_param[idx:idx + param.numel()] = torch.flatten(param).data.cpu()
                idx += param.numel()
    return flat_param


def flatten_grads(model, num_params, skip_bn_bias=False):
    flat_grads = torch.zeros(num_params, requires_grad=False)
    idx = 0
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() <= 1 and skip_bn_bias:
                pass
            else:
                flat_grads[idx:idx + param.numel()] = torch.flatten(param.grad).data.cpu()
                idx += param.numel()
    return flat_grads


def apply_params(model, array, skip_bn_bias=False):
    idx = 0
    for param in model.parameters():
        if param.requires_grad:
            if param.dim() <= 1 and skip_bn_bias:
                pass
            else:
                param.data = (array[idx:idx + param.numel()]).reshape(param.data.shape)
                idx += param.numel()
    return model
