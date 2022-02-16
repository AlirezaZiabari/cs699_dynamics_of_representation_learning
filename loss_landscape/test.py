import argparse
import torch

from loss_landscape.train import get_dataloader
from loss_landscape.utils.evaluations import get_loss_value_for_saved_data, get_loss_value

from utils.resnet import get_resnet, set_resnet_weights

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def test_model(model, device, use_saved_data=False, data_path=None, batch_size=1,
               attack_type=None, eps=None, alpha=None, iterations=None):
    model.eval()
    if use_saved_data:
        losses, accuracies = get_loss_value_for_saved_data(model, data_path, device)
    else:
        _, test_loader = get_dataloader(batch_size=batch_size)
        losses, accuracies = get_loss_value(model, test_loader, device,
                                            attack_type=attack_type, eps=eps, alpha=alpha, iterations=iterations)
    print(f"Test Accuracy: {accuracies}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # model related arguments
    parser.add_argument("--statefile", "-s", required=True, default=None)
    parser.add_argument(
        "--model", required=False, choices=["resnet20", "resnet32", "resnet44", "resnet56"], default="resnet20"
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument(
        "--skip_bn_bias", action="store_true", default=True,
        help="whether to skip considering bias and batch norm params or not, Li et al do not consider bias and batch norm params"
    )

    parser.add_argument("--use_saved_data", action="store_true", default=False)
    parser.add_argument("--data_path", required=False, default=None)
    parser.add_argument("--batch_size", required=False, type=int, default=1000)

    parser.add_argument("--attack_type", required=False, type=str, default="vanilla")

    parser.add_argument("--attack_eps", required=False, type=float, default=0.05)  # 2 for pgd_l2, 0.05 for pgd
    parser.add_argument("--attack_alpha", required=False, type=float, default=0.05)  # 0.2 for pgd_l2, 0.05 for pgd
    parser.add_argument("--attack_iters", required=False, type=int, default=20)

    args = parser.parse_args()

    model = get_resnet(args.model)(
        num_classes=10, remove_skip_connections=args.remove_skip_connections
    )
    state_dict = torch.load(args.statefile, map_location=args.device)
    model = set_resnet_weights(model, state_dict)

    test_model(model. args.device, args.use_saved_data, args.data_path, args.batch_size,
               args.attack_type, args.attack_eps, args.attack_alpha, args.attack_iters)
