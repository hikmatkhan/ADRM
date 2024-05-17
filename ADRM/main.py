import json
import argparse

import wandb

from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    _init_wandb(args)
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def _init_wandb(args):
    if args["wandb_log"]:
        wandb.init(project=args["wand_project"], entity=args["username"], reinit=True)
        wandb.config.update(args)





def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/reply.json',
                        help='Json file of settings.')
    parser.add_argument('--adversarial_robust_reply', type=int, default=-1)
    parser.add_argument('--adv_train', type=int, default=0)
    parser.add_argument('--adv_train_steps', type=int, default=5)
    parser.add_argument('--low_adv_epsilon', type=float, default=0.01)
    parser.add_argument('--high_adv_epsilon', type=float, default=0.09)
    parser.add_argument('--adv_attack', type=str, default="fgsm")
    parser.add_argument('--adv_attack_steps', type=int, default=5)


    # Replay Parameters
    parser.add_argument('--replay_EPSILON', type=float, default=1e-8)
    parser.add_argument('--replay_init_epoch', type=int, default=1)
    parser.add_argument('--replay_init_lr', type=float, default=0.1)
    parser.add_argument('--replay_init_lr_decay', type=float, default=0.1)
    parser.add_argument('--replay_init_weight_decay', type=float, default=0.0005)
    parser.add_argument('--replay_epochs', type=int, default=1)
    parser.add_argument('--replay_lrate', type=float, default=0.1)
    parser.add_argument('--replay_lrate_decay', type=float, default=0.1)
    parser.add_argument('--replay_batch_size', type=int, default=128)
    parser.add_argument('--replay_weight_decay', type=float, default=2e-4)
    parser.add_argument('--replay_T', type=int, default=2)


    #BIC parameters
    parser.add_argument('--bic_epochs', type=int, default=170)
    parser.add_argument('--bic_lrate', type=float, default=0.1)
    parser.add_argument('--bic_lrate_decay', type=float, default=0.1)
    parser.add_argument('--bic_batch_size', type=int, default=128)
    parser.add_argument('--bic_split_ratio', type=float, default=0.1)
    parser.add_argument('--bic_T', type=int, default=2)
    parser.add_argument('--bic_weight_decay', type=float, default=2e-4)

    # DER parameters
    parser.add_argument('--der_lrate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--der_lrate_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--der_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--der_T', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--der_init_lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--der_init_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--der_init_lr_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--der_init_epoch', type=int, default=2000, help='Learning rate.')
    parser.add_argument('--der_epochs', type=int, default=1700, help='Learning rate.')
    parser.add_argument('--der_batch_size', type=int, default=128, help='Learning rate.')

    # ICARL Parameters
    parser.add_argument('--icarl_init_epoch', type=int, default=200, help='Learning rate.')
    parser.add_argument('--icarl_epochs', type=int, default=250, help='Learning rate.')
    parser.add_argument('--icarl_lrate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--icarl_init_lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--icarl_lrate_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--icarl_init_lr_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--icarl_init_weight_decay', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--icarl_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--icarl_T', type=int, default=2, help='Learning rate.')
    parser.add_argument('--icarl_batch_size', type=int, default=128, help='Learning rate.')

    #PodNet
    parser.add_argument('--pd_epochs', type=int, default=200, help='Learning rate.')
    parser.add_argument('--pd_lrate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--pd_ft_epochs', type=int, default=20, help='Learning rate.')
    parser.add_argument('--pd_ft_lrate', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--pd_lambda_c_base', type=int, default=5, help='Learning rate.')
    parser.add_argument('--pd_lambda_f_base', type=int, default=1, help='Learning rate.')
    parser.add_argument('--pd_weight_decay', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--pd_nb_proxy', type=int, default=10, help='Learning rate.')
    parser.add_argument('--pd_batch_size', type=int, default=128, help='Learning rate.')

    # WA
    parser.add_argument('--wa_init_epoch', type=int, default=200, help='Learning rate.')
    parser.add_argument('--wa_init_lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--wa_init_lr_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--wa_init_weight_decay', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--wa_epochs', type=int, default=70, help='Learning rate.')
    parser.add_argument('--wa_lrate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--wa_lrate_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--wa_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--wa_T', type=int, default=2, help='Learning rate.')
    parser.add_argument('--wa_batch_size', type=int, default=128, help='Learning rate.')

    # Fetril
    parser.add_argument('--fetril_init_epochs', type=int, default=200, help='Learning rate.')
    parser.add_argument('--fetril_init_lr', type=float, default= 0.1, help='Learning rate.')
    parser.add_argument('--fetril_init_weight_decay', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--fetril_epochs', type=int, default=50, help='Learning rate.')
    parser.add_argument('--fetril_lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--fetril_batch_size', type=int, default=128, help='Learning rate.')
    parser.add_argument('--fetril_weight_decay', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--fetril_T', type=int, default=2, help='Learning rate.')

    # Foster Arguments

    parser.add_argument('--foster_init_epochs', type=int, default=200, help='Learning rate.')
    parser.add_argument('--foster_init_lr', type=float, default= 0.1, help='Learning rate.')
    parser.add_argument('--foster_init_weight_decay', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--foster_compression_epochs', type=int, default=130, help='Learning rate.')
    parser.add_argument('--foster_batch_size', type=int, default=128, help='Learning rate.')
    parser.add_argument('--foster_lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--foster_weight_decay', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--foster_boosting_epochs', type=int, default=170, help='Learning rate.')
    parser.add_argument('--foster_T', type=int, default=2, help='Learning rate.')

    # Pass Arguments
    parser.add_argument('--pass_temp', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--pass_epochs', type=int, default= 101, help='Learning rate.')
    parser.add_argument('--pass_lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--pass_batch_size', type=int, default=64, help='Learning rate.')
    parser.add_argument('--pass_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--pass_step_size', type=int, default=45, help='Learning rate.')
    parser.add_argument('--pass_gamma', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--pass_T', type=int, default=2, help='Learning rate.')

    # Memo Arguments
    parser.add_argument('--memo_init_epoch', type=int, default=200, help='Learning rate.')
    parser.add_argument('--memo_init_lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--memo_init_weight_decay', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--memo_init_lr_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--memo_epochs', type=int, default=170, help='Learning rate.')
    parser.add_argument('--memo_lrate', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--memo_batch_size', type=int, default=128, help='Learning rate.')
    parser.add_argument('--memo_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--memo_lrate_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--memo_T', type=int, default=2, help='Learning rate.')

    # Simple CIL
    parser.add_argument('--simplecil_init_epoch', type=int, default=128, help='Learning rate.')
    parser.add_argument('--simplecil_init_lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--simplecil_batch_size', type=int, default=256, help='Learning rate.')
    parser.add_argument('--simplecil_weight_decay', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--simplecil_init_lr_decay', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--simplecil_init_weight_decay', type=float, default=5e-4, help='Learning rate.')

    # IL2A

    parser.add_argument('--il2a_epochs', type=int, default=101, help='Learning rate.')
    parser.add_argument('--il2a_lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--il2a_batch_size', type=int, default=64, help='Learning rate.')
    parser.add_argument('--il2a_weight_decay', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--il2a_T', type=int, default=2, help='Learning rate.')




    parser.add_argument('--weight_path', type=str,
                        default="/data/hikmat/PGWorkspace/IJCNN2024",
                        help='Weight path')

    # Wandb visualization
    parser.add_argument('--wandb_log', type=int, default=1)
    parser.add_argument('--wand-project', type=str, default="PG_IJCNN2024", help='Project name.')
    parser.add_argument('--username', type=str, default="hikmatkhan-", help='Username')


    return parser


if __name__ == '__main__':
    main()
    exit()
