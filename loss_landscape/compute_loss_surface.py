"""
Here we compute loss at various points along the directions
We compute L(w+x*d1 + y*d2) at different values of x and y and save it to surface file which will be used to plot contours
"""
import argparse
from copy import deepcopy
import logging
import os
import sys
from unittest import skip

import dill
import numpy
import torch
from tqdm import tqdm

from train import get_dataloader
from utils.evaluations import get_loss_value, get_loss_value_for_saved_data
from utils.nn_manipulation import count_params, flatten_params, set_weights_by_direction, set_weights_from_swag_base
from utils.reproducibility import set_seed
from utils.resnet import get_resnet, set_resnet_weights
from swag.posteriors.swag import SWAG
from swag.utils import bn_update
torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model_folder", "-r", required=False) #, default=r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag_diag')
    parser.add_argument("--adversarial_folder", "-a", required=False)
    parser.add_argument("--statefile", "-s", required=False) #, default=r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag_diag\ckpt\200_model.pt')
    parser.add_argument(
        "--model", required=False, choices=["resnet20", "resnet32", "resnet44", "resnet56"], default='resnet20'
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument("--skip_bn_bias", action="store_true", default=True)

    parser.add_argument("--test_data", action="store_true", default=False)
    parser.add_argument("--use_attack", action="store_true", default=False)
    parser.add_argument("--data_root_dir", type=str, required=False)
    
    parser.add_argument("--use_swag_model", required=False, action='store_true', default=False)
    parser.add_argument("--use_swag_diag_cov", required=False, action='store_true', default=False)
    parser.add_argument("--swag_scale", required=False, type=float, default=1)
    parser.add_argument("--swag_num_samples", required=False, type=int, default=30)
    parser.add_argument("--swag_var_clamp", required=False, type=float, default=1e-5)

    parser.add_argument("--batch_size", required=False, type=int, default=1000)
    parser.add_argument("--direction_file", required=False, type=str) #, default=r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag_diag\pca_directions.npz')

    parser.add_argument(
        "--xcoords", type=str, default="5:-30:45",
        help="range of x-coordinate, specify as num:min:max"
    )
    parser.add_argument(
        "--ycoords", type=str, default="5:-30:45",
        help="range of y-coordinate, specify as num:min:max"
    )
    
    args = parser.parse_args()

    loss_surface_folder = f"{args.model_folder}/loss_surface/"
    # set up logging
    os.makedirs(f"{loss_surface_folder}", exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    set_seed(args.seed)

    # save name
    save_name = ''
    xlim_str = args.xcoords.replace(':', ',')

    if args.test_data:
        save_name += 'test'
    else:
        save_name += 'train'

    save_name += '_adversarial' if args.use_attack else '_vanilla'

    save_name += f'_{xlim_str}'

    if args.use_swag_model:
        save_name  += '_swag'
        if args.use_swag_diag_cov:
            save_name += '_diag'
        save_name += f'_{args.swag_num_samples}ns_{args.swag_scale:.1e}s_{args.swag_var_clamp:.1e}vc'
        
    if 'pca' in args.direction_file:
        save_name_buffer = f'pca_loss_surface_{save_name}.npz'
    elif 'last_1' in args.direction_file and 'last_10' not in args.direction_file:
        save_name_buffer = f'buffer_last_1_loss_surface_{save_name}.npz'
    elif 'last_10' in args.direction_file:
        save_name_buffer = f'buffer_last_10_loss_surface_{save_name}.npz'
    elif 'random' in args.direction_file:
        save_name_buffer = f'random_loss_surface_{save_name}.npz'
    else:
        save_name_buffer = f'buffer_loss_surface_{save_name}.npz'
    
        
    if os.path.exists(f"{loss_surface_folder}/{save_name_buffer}"):
        logger.error(f"{save_name_buffer} exists, so we will exit")
        sys.exit()

    # get dataset
    # using training dataset and only 10000 examples for faster evaluation
    train_loader, test_loader = get_dataloader(
        args.batch_size, transform_train_data=False, train_size=10000, get_adversarial=args.use_attack, root_dir=args.data_root_dir,
    )
    if args.test_data:
        loader = test_loader
    else:
        loader = train_loader

    # get model
    model = get_resnet(args.model)(num_classes=10, remove_skip_connections=args.remove_skip_connections)
    total_params = count_params(model, skip_bn_bias=args.skip_bn_bias)
    logger.info(f"using {args.model} with {total_params} parameters")   
    
    # load ckpt to model
    model_list = []
    pretrained_weights_list = []
    
    if args.use_swag_model:
        max_num_models = int(os.path.split(args.statefile)[-1].split('_')[0])
        swag_model = SWAG(get_resnet(args.model),
                    num_classes=10, remove_skip_connections=args.remove_skip_connections, 
                    no_cov_mat=args.use_swag_diag_cov, max_num_models=max_num_models, var_clamp=args.swag_var_clamp)
        
        logger.info(f"Loading SWAG model from {args.statefile}")
        state_dict = torch.load(args.statefile, pickle_module=dill, map_location=args.device)
        swag_model.load_state_dict(state_dict)
        swag_model.to(args.device)
        
        # update batchnorm statistics
        swag_model.sample(0.0)
        bn_update(loader, swag_model, verbose=True, subset=0.1)

        # get swag sampled models
        for model_id in range(args.swag_num_samples):
            swag_model.sample(scale=args.swag_scale, cov=not args.use_swag_diag_cov)
            bn_update(loader, swag_model, verbose=True, subset=0.1)
            
            set_weights_from_swag_base(model, swag_model.base) 
            model_list.append(deepcopy(model)) 
            
            pretrained_weights_list.append(flatten_params(model_list[model_id], num_params=total_params, skip_bn_bias=args.skip_bn_bias).to(args.device))
    else:
        model.to(args.device)   
        
        # load ckpt to model
        logger.info(f"Loading model from {args.statefile}")
        state_dict = torch.load(args.statefile, pickle_module=dill, map_location=args.device)
        model = set_resnet_weights(model, state_dict)
        
        model_list.append(model)
        pretrained_weights_list.append(flatten_params(model, num_params=total_params, skip_bn_bias=args.skip_bn_bias).to(args.device))
        
    logger.info(f"Loading directions from {args.direction_file}")
    temp = numpy.load(args.direction_file)
    direction1 = torch.tensor(temp["direction1"], device=args.device).float()
    direction2 = torch.tensor(temp["direction2"], device=args.device).float()

    x_num, x_min, x_max = [float(i) for i in args.xcoords.split(":")]
    y_num, y_min, y_max = [float(i) for i in args.ycoords.split(":")]

    x_num, y_num = int(x_num), int(y_num)

    logger.info(f"x-range: {x_min}:{x_max}:{x_num}")
    logger.info(f"y-range: {y_min}:{y_max}:{y_num}")

    xcoordinates = numpy.linspace(x_min, x_max, num=x_num)
    ycoordinates = numpy.linspace(y_min, y_max, num=y_num)

    losses = numpy.zeros((x_num, y_num))
    accuracies = numpy.zeros((x_num, y_num))

    with tqdm(total=x_num * y_num) as pbar:
        for idx_x, x in enumerate(xcoordinates):
            for idx_y, y in enumerate(ycoordinates):
                for model_id in range(len(model_list)):
                    
                    set_weights_by_direction(model_list[model_id], x, y, direction1, direction2, pretrained_weights_list[model_id],
                                            skip_bn_bias=args.skip_bn_bias)
                    model_list[model_id].to('cpu')

                losses[idx_x, idx_y], accuracies[idx_x, idx_y] = get_loss_value(model_list, loader, args.device)

                pbar.set_description(f"x:{x: .4f}, y:{y: .4f}, loss:{losses[idx_x, idx_y]:.4f}")
                pbar.update(1)

    # save losses and accuracies evaluations
    logger.info("Saving results")
    numpy.savez(
        f"{loss_surface_folder}/{save_name_buffer}", losses=losses, accuracies=accuracies,
        xcoordinates=xcoordinates, ycoordinates=ycoordinates
    )
