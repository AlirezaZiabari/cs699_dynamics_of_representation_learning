import argparse
import torch

from train import get_dataloader
from utils.evaluations import get_loss_value
from utils.nn_manipulation import set_weights_from_swag_base

from utils.resnet import get_resnet, set_resnet_weights

import ssl
import copy
from swag.posteriors.swag import SWAG
from swag.utils import bn_update
from adversarial_attack import load_swag_models
from utils import swag_config

ssl._create_default_https_context = ssl._create_unverified_context


def test_model(model, device, data_root_dir=None, batch_size=1, use_adversarial_saved_data=False, 
               attack_type=None, eps=None, alpha=None, iterations=None, train_data=False):
    model.eval()
    
    train_loader, test_loader = get_dataloader(batch_size=batch_size, root_dir=data_root_dir, get_adversarial=use_adversarial_saved_data)
    
    if not train_data:
        loader = test_loader
    else:
        loader = train_loader 
        
    _, accuracy = get_loss_value([model], loader, device,
                                        attack_type=attack_type, eps=eps, alpha=alpha, iterations=iterations)
    print(f"Test Accuracy: {accuracy}")

    
def test_swag_model(swag_model, model, device, batch_size=1, 
                    num_samples=10, scale=1, diag_cov=True,
                    data_root_dir=None, use_adversarial_saved_data=False, train_data=False, swag_samples_path=None):
    '''
    Gets samples from SWAG model and tests it on either vanilla data or pre-saved adversarial data. 
    Adversarial data is saved by using save_adversarial_images(...) function under load_data.py, the trained model from final epoch is used
    when creating adversarial images and same adversarial images are used for all SWAG models. 
    It means that adversarial images are not created for each SWAG model separately. 
    
    Parameters
    ---------------
    swag_model: SWAG model to get samples from
    model: default model which SWAG uses as its base
    device: which device to do operations on
    batch_size: batch size 
    num_samples: number of samples to get from SWAG model
    scale: covariance scale for SWAG model
    diag_cov: whether to use diagonal or full covariance when taking the samples from SWAG model
    data_root_dir: directory where the adversarial data is saved, e.g. {data_root_dir}/train(test)/adversarial_image_1.pt, used only if use_adversarial_saved_data is True
    use_adversarial_saved_data: whether to use saved adversarial saved data (only option if adversarial test performance is desired to be obtained)
    test_data: whether to use test or train dataloader for printing accuracy 
    
    Prints (No returns)
    ------------
    Prints the SWAG cumulative test performance either on vanilla or adversarial data
    '''
    
    train_loader, test_loader = get_dataloader(batch_size=batch_size, root_dir=data_root_dir, get_adversarial=use_adversarial_saved_data)
    
    if not train_data:
        loader = test_loader
    else:
        loader = train_loader 
        
    # update batchnorm statistics
    swag_model.sample(0.0)
    bn_update(loader, swag_model, verbose=True, subset=0.1)
    
    model_list = []
    if swag_samples_path:
        model_list = load_swag_models(swag_samples_path, model, device)
    else:
        for _ in range(num_samples):
            # sample swag model and update batchnorm statistics
            swag_model.sample(scale=scale, cov=not diag_cov)
            bn_update(loader, swag_model, verbose=True, subset=0.1)
            
            set_weights_from_swag_base(model, swag_model.base)
            model_list.append(copy.deepcopy(model))
    
    
    _, accuracy = get_loss_value(model_list, loader, device)
    
    print(f"Test Accuracy: {accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # model related arguments
    parser.add_argument("--result_folder", required=False) #, default=r'C:\Users\berkt\Desktop\cs699_dynamics_of_representation_learning\loss_landscape\results\resnet20_skip_bn_bias_swag_diag')
    parser.add_argument("--ckpt_load", required=False, type=int, default=200)
    parser.add_argument("--model", required=False, choices=["resnet20", "resnet32", "resnet44", "resnet56"], default="resnet20")
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument(
        "--skip_bn_bias", action="store_true", default=True,
        help="whether to skip considering bias and batch norm params or not, Li et al do not consider bias and batch norm params"
    )

    parser.add_argument("--data_root_dir", required=False, default=None)
    parser.add_argument("--use_adversarial_saved_data", action="store_true", default=False)
    parser.add_argument("--train_data", action="store_true", default=False)
    parser.add_argument("--batch_size", required=False, type=int, default=256)

    parser.add_argument("--attack_type", required=False, type=str)
    parser.add_argument("--attack_eps", required=False, type=float, default=0.05)  # 2 for pgd_l2, 0.05 for pgd
    parser.add_argument("--attack_alpha", required=False, type=float, default=0.05)  # 0.2 for pgd_l2, 0.05 for pgd
    parser.add_argument("--attack_iters", required=False, type=int, default=20)
    
    parser.add_argument("--use_swag_model", required=False, action='store_true', default=False)
    parser.add_argument("--swag_samples_path", required=False, type=str)
    
    parser.add_argument

    args = parser.parse_args()
    
    if not args.attack_type:
        torch.set_grad_enabled(False)
    
    if args.use_swag_model:
        model = get_resnet(args.model)(
            num_classes=10, remove_skip_connections=args.remove_skip_connections
        )
        
        swag_model = SWAG(get_resnet(args.model), 
                          num_classes=10, remove_skip_connections=args.remove_skip_connections, 
                          no_cov_mat=swag_config.USE_DIAG_COV, max_num_models=args.ckpt_load, var_clamp=swag_config.VAR_CLIP)
        
        load_path = f"{args.result_folder}/swag_ckpt/{args.ckpt_load}_swag_model"
        if swag_config.USE_DIAG_COV:
            load_path += '_diag'
            
        state_dict = torch.load(f"{load_path}.pt", map_location=args.device)
        swag_model.load_state_dict(state_dict)
        swag_model.to(args.device)
        test_swag_model(swag_model=swag_model, model=model, device=args.device, data_root_dir=args.data_root_dir, batch_size=args.batch_size, 
                        num_samples=swag_config.NUM_SAMPLES, scale=swag_config.SCALE, diag_cov=swag_config.USE_DIAG_COV,
                        use_adversarial_saved_data=args.use_adversarial_saved_data, train_data=args.train_data, swag_samples_path=args.swag_samples_path)
        
    else:
        model = get_resnet(args.model)(
            num_classes=10, remove_skip_connections=args.remove_skip_connections
        )
        
        state_dict = torch.load(f"{args.result_folder}/ckpt/{args.ckpt_load}_model.pt", map_location=args.device)     
        model = set_resnet_weights(model, state_dict)
        model.to(args.device)
        test_model(model=model, device=args.device, data_root_dir=args.data_root_dir, batch_size=args.batch_size,
                   attack_type=args.attack_type, eps=args.attack_eps, alpha=args.attack_alpha, iterations=args.attack_iters,
                   use_adversarial_saved_data=args.use_adversarial_saved_data, train_data=args.train_data)

