import sys
import logging
import copy

import numpy as np
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, tensor2numpy
import os



def save_at(obj, path):
    torch.save(obj, "{}.pkl".format(path))

def _get_weight_fpath(args):
    # WEIGHT_PATH http://localhost:8080/tree/data/hikmat/RUWorkspace/PYCIL/PyCIL

    # WEIGHT_PATH = "/data/hikmat/RUWorkspace/PYCIL_L2/ORACLE_WEIGHTS"
    # WEIGHT_PATH = "/data/hikmat/PGWorkspace/IJCNN2024"
    if(args["adv_train"]):
        path = "{}/_Adv_ROBUST_WEIGHTS_{}".format(args["weight_path"], args["normalize"].upper())
    else:
        path = "{}/_WEIGHTS_{}".format(args["weight_path"], args["normalize"].upper())
    # path = "{}/_WEIGHTS_{}".format(WEIGHT_PATH, args["normalize"].upper())

    if not os.path.exists(path):
        os.mkdir(path)

    path = "{}/{}".format(path, args["dataset"])
    if not os.path.exists(path):
        os.mkdir(path)

    path = "{}/{}_{}_{}_M_{}_MPC_{}".format(path, args["dataset"], args["init_cls"], args["increment"],
                                            args["memory_size"], args["memory_per_class"])
    if not os.path.exists(path):
        os.mkdir(path)

    path = "{}/{}".format(path, args["model_name"])
    if not os.path.exists(path):
        os.mkdir(path)

    path = "{}/{}".format(path, args["convnet_type"])
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def _log_on_wandb(args, log_dict):
    if args["wandb_log"]:
        wandb.log(log_dict)


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def visualize(imgs, labels, task, row=8, cols=8):
    # imgs = imgs.transpose(0, 2, 3, 1)
    # now we need to unnormalize our images.
    fig = plt.figure(figsize=(8, 8))
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(row, cols, i + 1, xticks=[], yticks=[])
        ax.imshow(imgs[i])
        ax.set_title(labels[i].item(), fontsize=10)
    plt.show()
    plt.savefig('/home/khanhi83/RUWorkspace/PGWorkspace/PyCIL/test_images/{}_{}.jpg'.format(task, np.unique(labels)[0]))


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    weight_path = _get_weight_fpath(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top{}".format(args["topk"]): []}, {"top1": [],
                                                                            "top{}".format(args["topk"]): []}
    for task in range(data_manager.nb_tasks):
        # if model._get_memory() is None:
        #     logging.info("==> Learning task # {} with memory buffer size={}".format(task, 0))
        # else:
        #     logging.info("==> Learning task # {} with memory buffer size={}".format(task, len(model._get_memory()[1])))
        #     data, targets = model._get_memory()
        #     print("Targets:", type(targets), " ", targets.shape, " Unique:", np.unique(targets))
        #     for i in np.unique(targets):
        #         indices = np.where(targets == i)
        #         print(
        #             "==> Class {} has {} samples available.".format(np.unique(targets[indices]), len(targets[indices])))
        #
        #         visualize(data[indices][:64], targets[indices][:64], str(task), row=8, cols=8)
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        # continue
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()

        print("*" * 100)
        print("CNN_ACCY=", cnn_accy)
        print("==>Head was:", model._network.fc.out_features)
        print("*" * 100)


        if args["save_model"] is True:
            if args["model_name"] == "memo":
                saved_w_path = model.save_memo_checkpoint(w_path=weight_path, test_acc=cnn_accy)
                # pass
            else:
                saved_w_path = model.save_checkpoint(w_path=weight_path)

        # print("*" * 100)
        # model.load_task_checkpoint(w_path=saved_w_path)
        # cnn_accy, nme_accy = model.eval_task()
        # print("==> CNN_ACCY with loaded weights=", cnn_accy)
        # print("*" * 100)

        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top{}".format(args["topk"])].append(cnn_accy["top{}".format(args["topk"])])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top{}".format(args["topk"])].append(nme_accy["top{}".format(args["topk"])])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top{} curve: {}".format(args["topk"], cnn_curve["top{}".format(args["topk"])]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top{} curve: {}\n".format(args["topk"], nme_curve["top{}".format(args["topk"])]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))

            cnn_avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            nme_avg_acc = sum(nme_curve["top1"]) / len(nme_curve["top1"])
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
            _log_on_wandb(args=args, log_dict={"cnn_avg_acc": cnn_avg_acc, "nme_avg_acc": nme_avg_acc})

            save_at(cnn_accy["grouped"], "{}/{}_cnn_accy.pkl".format(weight_path, task))
            save_at(nme_accy["grouped"], "{}/{}_nme_accy.pkl".format(weight_path, task))

        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top{}".format(args["topk"])].append(cnn_accy["top{}".format(args["topk"])])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top{} curve: {}\n".format(args["topk"], cnn_curve["top{}".format(args["topk"])]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
            cnn_avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            _log_on_wandb(args=args, log_dict={"cnn_avg_acc": cnn_avg_acc})
            save_at(cnn_accy["grouped"], "{}/{}_cnn_accy.pkl".format(weight_path, task))
    print("Saving At:", "{}/cnn_curve_avg_acc.pkl".format(weight_path))
    # torch.save(nme_curve, "{}/nme_curve_avg_acc.pkl".format(weight_path))
    save_at(obj=nme_curve, path="{}/nme_curve_avg_acc.pkl".format(weight_path))
    # _log_avg_acc(args=args, metric="nm", method=args["model_name"], avg_acc=nme_curve)
    save_at(obj=cnn_curve, path="{}/cnn_curve_avg_acc.pkl".format(weight_path))
    # _log_avg_acc(args=args, metric="cn", method=args["model_name"], avg_acc=cnn_curve)
    try:
        if args["perform_cifar10_c_eval"]:
            perform_adv_evaluation(agent=model, pycil_args=args)
            perform_evaluation_on_cifar10c(agent=model, args=args,
                                           val_acc=sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
        else:
            print("==> NO CIFAR10-C Evaluation")
    except Exception as e:
        print(e)



def load_cifar10(path, corruptions="fog", severity="clean"):
    # sample = torch.from_numpy(np.load(path))
    start_indx = (severity * 10000) - 10000
    end_indx = severity * 10000
    #     x, y = (torch.from_numpy(np.load("./{0}.npy".format(corruptions)))[start_indx: end_indx],
    #             torch.from_numpy(np.load("./labels.npy"))[start_indx: end_indx])
    x, y = (torch.from_numpy(np.load("{}/{}.npy".format(path, corruptions)))[start_indx: end_indx],
            torch.from_numpy(np.load("{0}/labels.npy".format(path)))[start_indx: end_indx])
    x, y = x.permute(0, 3, 1, 2), y
    #     print("X:::", x.shape, " y:::", y.shape)

    #     normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
    #                                  std=[0.247, 0.243, 0.262])
    # Transformation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        #         normalize,

    ])
    dataset = CCIFARTensorDataset(x, y, transform=val_transform)
    #     print("No of classes:", torch.unique(y))
    #     trainloader = DataLoader(dataset, batch_size=64)
    #     print(x.shape, ",", y.shape)
    return dataset

from torch.utils.data import TensorDataset, Dataset

from torch.utils.data import TensorDataset, Dataset

class CCIFARTensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        #         print("Img Before:", im.shape)
        #         print("Target:", targ)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        #             print("Img After:", im.shape)
        return im, int(targ)

    def __len__(self):
        return self.tensors[0].size(0)


def get_adv_dataset(eps, attack, path="/data/hikmat/Adv_CIFAR10/"):
    import torch
    # epss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192,
    #         224]
    print("{}-{}-{}".format(attack, eps, path))
    x = torch.load("{}/{}_{}_x.pt".format(path, attack, eps))

    # if attack == "pgd_l2":
    #     y = torch.load("{}/{}_y.pt".format(path, epss[i]))
    # else:
    y = torch.load("{}/{}_{}_y.pt".format(path, attack, eps))


    return x, y
def perform_adv_evaluation(agent, pycil_args):
    if ("not" in pycil_args["normalize"]):
        cl_val_transform = transforms.Compose([
            transforms.ToTensor()

        ])
        print("NOT NORMALIZED INPUT...")
    else:
        cl_val_transform = transforms.Compose([
            transforms.ToTensor()#,
            # transforms.Normalize(
            #     mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            # )
        ])
        print("NORMALIZED INPUT...")
    cl_val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=cl_val_transform)
    #
    cl_val_loader = torch.utils.data.DataLoader(cl_val_dataset, batch_size=256, shuffle=False, num_workers=4)
    val_acc = agent._cl_compute_accuracy(model=agent._network, loader=cl_val_loader, device=pycil_args["device"][0])

    fgsm_accuracies = []
    pgd_linf_accuracies = []
    pdg_l2_accuracies = []
    fgsm_accuracies.append(val_acc)
    pgd_linf_accuracies.append(val_acc)
    pdg_l2_accuracies.append(val_acc)

    ATTACKS = [
        {"fgsm": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192, 224]},
        {"pgd_linf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192, 224]},
        {"pgd_l2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.0, 12.,
                    14., 16.]}
    ]

    for ATCK in ATTACKS:
        for attack, epss in ATCK.items():

            for eps in epss:
                x, y = get_adv_dataset(eps=eps, attack=attack)
                adv_dataset = CCIFARTensorDataset(x, y, transform=cl_val_transform)
                print("X:", x.shape, "\t Y:", y.shape, "\t X:", type(x), "\t Y:", type(y))
                print("Evaluating on {} with epsilon {}".format(attack, eps))
                adv_val_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1024, shuffle=False)
                print("dataset:", len(adv_dataset), "\t cl_val_loader=", len(adv_val_loader))
                if attack == "fgsm":
                    fgsm_accuracies.append(agent._adv_compute_accuracy(model=agent._network,
                                                                       loader=adv_val_loader, device=pycil_args["device"][0]))
                elif attack == "pgd_linf":
                    pgd_linf_accuracies.append(agent._adv_compute_accuracy(model=agent._network,
                                                                           loader=adv_val_loader, device=pycil_args["device"][0]))
                elif attack == "pgd_l2":
                    pdg_l2_accuracies.append(agent._adv_compute_accuracy(model=agent._network,
                                                                         loader=adv_val_loader, device=pycil_args["device"][0]))
                else:
                    print("Error: Shouldn't be the case.")
        #         break
        # break

    print("*" * 100)
    print(fgsm_accuracies)
    print("*" * 100)
    print(pgd_linf_accuracies)
    print("*" * 100)
    print(pdg_l2_accuracies)
    print("*" * 100)
    torch.save({"fgsm": fgsm_accuracies, "pgd_linf": pgd_linf_accuracies,
                "pgd_l2": pdg_l2_accuracies},
               "/data/hikmat/PGWorkspace/IJCNN2024/Evaluation/Adv_{}-C_{}_Adv_{}_M_{}_{}_{}.pt".format(
                   pycil_args["dataset"], pycil_args["model_name"], pycil_args["use_adv_weight"],
                   pycil_args["memory_size"], pycil_args["init_cls"], pycil_args["increment"]))


def perform_evaluation_on_cifar10c(agent, val_acc, args):

    corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                   "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                   "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
    CIFAR10_C_Path = "/home/khanhi83/ACLWorkspace/CIFAR-10-C"
    robust_acc = {}
    for c in corruptions:
        r_acc = []
        r_acc.append(val_acc)
        for s in range(1, 6):

            c_loader = DataLoader(load_cifar10(path=CIFAR10_C_Path, corruptions=c, severity=s), batch_size=1024)
            t_acc = agent._adv_compute_accuracy(model=agent._network,
                                                                         loader=c_loader, device=args["device"][0])
            print("{} with severity {} obtain acc {}".format(c, s, t_acc))
            r_acc.append(t_acc)
        robust_acc[c] = r_acc
        # break
    from pprint import pprint
    pprint(robust_acc)
    torch.save(robust_acc, "/data/hikmat/PGWorkspace/IJCNN2024/Evaluation/{}-C_{}_Adv_{}_M_{}_{}_{}.pt".format(args["dataset"], args["model_name"],args["use_adv_weight"],
                                                                                         args["memory_size"], args["init_cls"], args["increment"]))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
