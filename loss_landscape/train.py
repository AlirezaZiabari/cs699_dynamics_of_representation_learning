"""
Script to train a neural network:
    currently supports training resnet for CIFAR-10 with and w/o skip connections

    Also does additional things that we may need for visualizing loss landscapes, such as using
      frequent directions or storing models during the executions etc.
   This has limited functionality or options, e.g., you do not have options to switch datasets
     or architecture too much.
"""

import argparse
import logging
import os
import pprint
import time
import json

import dill
import numpy.random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from utils.evaluations import get_loss_value
from utils.linear_algebra import FrequentDirectionAccountant
from utils.nn_manipulation import count_params, flatten_grads
from utils.reproducibility import set_seed
from utils.resnet import get_resnet, set_resnet_weights
import matplotlib.pyplot as plt

from adversarial_attack import pgd_attack_l2, pgd_attack
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# data folder
DATA_FOLDER = "../data/"


def get_dataloader(batch_size, train_size=None, test_size=None, transform_train_data=True):
    """
        returns: cifar dataloader

    Arguments:
        batch_size:
        train_size: How many samples to use of train dataset?
        test_size: How many samples to use from test dataset?
        transform_train_data: If we should transform (random crop/flip etc) or not
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize
        ]
    ) if transform_train_data else transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])


    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_FOLDER, train=False, transform=test_transform, download=True
    )

    
    if train_size:
        indices = numpy.random.permutation(numpy.arange(len(train_dataset)))
        train_dataset = Subset(train_dataset, indices[:train_size])

    if test_size:
        indices = numpy.random.permutation(numpy.arange(len(test_dataset)))
        test_dataset = Subset(train_dataset, indices[:test_size])

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--result_folder", "-r", required=False, default="C:/Users/berkt/Desktop/cs699_dynamics_of_representation_learning/loss_landscape/results_final/resnet20_skip_bn_bias")

    # model related arguments
    parser.add_argument("--statefile", "-s", required=False, default=None)
    parser.add_argument(
        "--model", required=False, choices=["resnet20", "resnet32", "resnet44", "resnet56"], default="resnet20"
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument("--use_adversarial_training", action="store_true", default=True)
    parser.add_argument("--test_adv", action="store_true", default=False)
    parser.add_argument(
        "--skip_bn_bias", action="store_true", default=True,
        help="whether to skip considering bias and batch norm params or not, Li et al do not consider bias and batch norm params"
    )

    parser.add_argument("--batch_size", required=False, type=int, default=1000)
    parser.add_argument(
        "--save_strategy", required=False, nargs="+", choices=["epoch", "init"],
        default=["epoch", "init"]
    )

    parser.add_argument("--test_model", action="store_true", default=True)
    parser.add_argument("--ckpt_load", required=False, type=int, default=200)
    parser.add_argument("--attack_type", required=False, type=str, default="pgd")
    
    parser.add_argument("--attack_eps", required=False, type=float, default=0.05) # 2 for pgd_l2, 0.05 for pgd
    parser.add_argument("--attack_alpha", required=False, type=float, default=0.05) # 0.2 for pgd_l2, 0.05 for pgd
    parser.add_argument("--attack_iters", required=False, type=int, default=20)

    parser.add_argument("--num_epochs", required=False, type=int, default=200)
    parser.add_argument("--lr", required=False, type=float, default=0.1)
    

    args = parser.parse_args()
    print('--------------------------------------')
    print('Training will start with following arguments:')
    print(args)
    print('GPU Info: {}, {}'.format(torch.cuda.device_count(), torch.cuda.is_available()))
    print('--------------------------------------')

    # set up logging
    os.makedirs(f"{args.result_folder}/ckpt", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    summary_writer = SummaryWriter(log_dir=args.result_folder)
    logger.info("Config:")
    logger.info(pprint.pformat(vars(args), indent=4))

    # save config file as config.json
    with open(f"{args.result_folder}/config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)
    
    # sets the seed
    set_seed(args.seed)

    # get dataset
    read_dir = "C:/Users/berkt/Desktop/cs699_dynamics_of_representation_learning/loss_landscape/results_final/resnet20_skip_bn_bias/"
    print(f'Reading pgd images from: {read_dir}')
    # train_loader, test_loader = get_dataloader(args.batch_size, use_adversarial=True, dir=read_dir, save_name="pgd_image_")
    train_loader, test_loader = get_dataloader(args.batch_size)

    # get model
    model = get_resnet(args.model)(
        num_classes=10, remove_skip_connections=args.remove_skip_connections
    )

    if torch.cuda.device_count() > 1:
        model= torch.nn.DataParallel(model)


    model.to(args.device)
    logger.info(f"using {args.model} with {count_params(model)} parameters")

    logger.debug(model)

    # we can try computing principal directions from some specific training rounds only
    total_params = count_params(model, skip_bn_bias=args.skip_bn_bias)
    fd = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    # frequent direction for last 10 epoch
    fd_last_10 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    # frequent direction for last 1 epoch
    fd_last_1 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)

    # use the same setup as He et al., 2015 (resnet)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda x: 1 if x < 100 else (0.1 if x < 150 else 0.01)
    )

    if "init" in args.save_strategy:
        torch.save(
            model.state_dict(), f"{args.result_folder}/ckpt/init_model.pt", pickle_module=dill
        )

    if not args.test_model:
        if args.ckpt_load > 0:        
            # model= torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(f"{args.result_folder}/ckpt/{args.ckpt_load}_model.pt", map_location=args.device))
        
        # training loop
        # we pass flattened gradients to the FrequentDirectionAccountant before clearing the grad buffer
        total_step = len(train_loader) * args.num_epochs
        step = 0
        direction_time = 0
        
        for epoch in range(args.num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(args.device)
                labels = labels.to(args.device)

                # Adversarial attack to images
                if args.attack_type == 'pgd_l2':
                    images_adv = pgd_attack_l2(model, images, labels, eps=args.attack_eps, alpha=args.attack_alpha, iters=args.attack_iters, device=args.device)
                elif args.attack_type == 'pgd':
                    images_adv = pgd_attack(model, images, labels, eps=args.attack_eps, alpha=args.attack_alpha, iters=args.attack_iters, device=args.device)
                else:
                    assert 1==0, 'Only available attack types are PGD and PGD-L2!'
                    
                # Forward pass, vanilla
                outputs = model(images)
                train_loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                # Forward pass, adversarial
                outputs_adv = model(images_adv)
                train_loss_adv = torch.nn.functional.cross_entropy(outputs_adv, labels)
                    
                # Backward and optimize, on adversarial
                optimizer.zero_grad()
                if args.use_adversarial_training:
                    train_loss_adv.backward()
                else:
                    train_loss.backward()
                    
                optimizer.step()

                # get gradient and send it to the accountant
                start = time.time()
                fd.update(flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias))
                direction_time += time.time() - start

                if epoch >= args.num_epochs - 10:
                    fd_last_10.update(
                        flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
                    )
                if epoch >= args.num_epochs - 1:
                    fd_last_1.update(
                        flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
                    )

                summary_writer.add_scalar("train/loss_adv", train_loss_adv.item(), step)
                summary_writer.add_scalar("train/loss", train_loss.item(), step)
                step += 1

                if step % 100 == 0:
                    logger.info(
                        f"Epoch [{epoch}/{args.num_epochs}], Step [{step}/{total_step}] Train Loss: {train_loss.item():.4f}, Train Adversarial Loss: {train_loss_adv.item():.4f}"
                    )

            scheduler.step()
            
            # Save the model checkpoint
            if "epoch" in args.save_strategy:
                save_name = epoch + 1
                if args.ckpt_load > 0:
                    save_name += args.ckpt_load 
                    
                torch.save(
                    model.state_dict(), f'{args.result_folder}/ckpt/{save_name}_model.pt',
                    pickle_module=dill
                )

            loss, acc = get_loss_value(model, test_loader, device=args.device, attack_type='vanilla')
            logger.info(f'Accuracy of the model on the vanilla test images: {100 * acc}%')
            summary_writer.add_scalar("test/acc", acc, step)
            summary_writer.add_scalar("test/loss", loss, step)
            
            if args.test_adv:
                loss_adv, acc_adv = get_loss_value(model, test_loader, device=args.device,
                                                              attack_type=args.attack_type, attack_eps=args.attack_eps, 
                                                              attack_alpha=args.attack_alpha, attack_iters=args.attack_iters)
                logger.info(f'Accuracy of the model on the adversarial test images: {100 * acc_adv}%')
                summary_writer.add_scalar("test/acc_adv", acc_adv, step)
                summary_writer.add_scalar("test/loss_adv", loss_adv, step)
            

        logger.info(f"Time to computer frequent directions {direction_time} s")

        logger.info(f"fd was updated for {fd.step} steps")
        logger.info(f"fd_last_10 was updated for {fd_last_10.step} steps")
        logger.info(f"fd_last_1 was updated for {fd_last_1.step} steps")

        # save the frequent_direction buffers and principal directions
        buffer = fd.get_current_buffer()
        directions = fd.get_current_directions()
        directions = directions.cpu().data.numpy()

        numpy.savez(
            f"{args.result_folder}/buffer.npy",
            buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
        )

        # save the frequent_direction buffer
        buffer = fd_last_10.get_current_buffer()
        directions = fd_last_10.get_current_directions()
        directions = directions.cpu().data.numpy()

        numpy.savez(
            f"{args.result_folder}/buffer_last_10.npy",
            buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
        )

        # save the frequent_direction buffer
        buffer = fd_last_1.get_current_buffer()
        directions = fd_last_1.get_current_directions()
        directions = directions.cpu().data.numpy()

        numpy.savez(
            f"{args.result_folder}/buffer_last_1.npy",
            buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
        )

    else:
       
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
        
        # load saved model
        state_dict = torch.load(f"{args.result_folder}/ckpt/{args.ckpt_load}_model.pt", map_location=device)
        model = set_resnet_weights(model, state_dict)

        model.eval()
        pred_labels = []
        pred_labels_pgd = []
        pred_labels_pgd_l2 = []
        
        labels_all = []
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

        for images, labels in test_loader:
            
            images = images.to("cuda:0")
            
            pred_labels_batch = model(images).argmax(axis=1).detach().to('cpu')
            
            images_pgd = pgd_attack(model, images, labels, eps=0.05, alpha=0.05, iters=20)
            
            images_inv = invTrans(images)
            images_pgd_inv = invTrans(images_pgd)
            
             
            # images_pgd_l2 = pgd_attack_l2(model, images, labels, iters=20, eps=2, alpha=0.1)
            
            # pred_labels_pgd_batch = model(images_pgd).argmax(axis=1).detach().to('cpu')
            # pred_labels_pgd_l2_batch = model(images_pgd_l2).argmax(axis=1).detach().to('cpu')

            pred_labels.append(pred_labels_batch)
            # pred_labels_pgd.append(pred_labels_pgd_batch)
            # pred_labels_pgd_l2.append(pred_labels_pgd_l2_batch)
            labels_all.append(labels)

        pred_labels = torch.cat(pred_labels)
        # pred_labels_pgd =  torch.cat(pred_labels_pgd)
        # pred_labels_pgd_l2 =  torch.cat(pred_labels_pgd_l2)
        labels_all =  torch.cat(labels_all)
        
        acc = (labels_all == pred_labels).float().mean().cpu().data.numpy()
        # acc_pgd = (labels_all == pred_labels_pgd).float().mean().cpu().data.numpy()
        # acc_pgd_l2 = (labels_all == pred_labels_pgd_l2).float().mean().cpu().data.numpy()
        
        # print(f"Test Accuracy on \n\
        #         vanilla images: {acc}, \n\
        #         on pgd images: {acc_pgd} \n")
        print(f"Test Accuracy on \n\
                vanilla images: {acc}")