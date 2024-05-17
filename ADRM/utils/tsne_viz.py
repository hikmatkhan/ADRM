import argparse
from tsnecuda import TSNE
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# from feature_extractor import get_architecture, load_weights, validate1, validate2, get_model, get_feature_extractor
# from pytorchcifar.models import RegNetX_200MF, VGG, PreActResNet18, GoogLeNet, ResNeXt29_2x64d, MobileNet, MobileNetV2, \
#     DPN92, ShuffleNetG2, SENet18, ShuffleNetV2, EfficientNetB0, SimpleDLA, ResNet18, ResNet34

parser = argparse.ArgumentParser(description='CL Dataset Creation')
parser.add_argument('--gpuid', type=int, default=5,
                    help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
parser.add_argument('--weight-path', type=str,
                    # default="/data/hikmat/RUWorkspace/RUCDLCreation/pytorchcifar/checkpoint",
                    # "/home/khanhi83/RUWorkspace/Weights/WideResNet_28_2_cifar",
                    default="/data/hikmat/PGWorkspace/IJCNN2024/EmbeddedFeatures",
                    help="")

parser.add_argument('--num-classes', type=int, default=10,
                    help='output classes.')
parser.add_argument('--model-name', type=str, default="resnet32",  # "ResNet18",
                    help='output classes.')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='Dataset name.')

# parser.add_argument('--agent', type=str, default="replay",
#                     help='replay or wa etc.')
parser.add_argument('--use-adv-weight', type=int, default=0,
                    help='1=Yes, -1=No')
# Run the cl cl and t and cl
parser.add_argument('--x-ds', type=str, default="t",
                    help='1=Yes, -1=No')
parser.add_argument('--cl-ds', type=str, default="t",
                    help='1=Yes, -1=No')

parser.add_argument('--config', type=str, default="./PYCIL/exps/wa.json",
                    help='Json file of settings.')
parser.add_argument('--root-weight-folder', type=str,
                    default="/data/hikmat/PGWorkspace/IJCNN2024",
                    # default="/data/hikmat/RUWorkspace/RUCDLCreation",
                    help='Path to weights folder')

parser.add_argument('--tsne_path', type=str, default="/data/hikmat/IJCNN2024",
                    help='Json file of settings.')

parser.add_argument('--num_neighbors', type=int, default=128, help='Json file of settings.')
parser.add_argument('--perplexity', type=int, default=50, help='Json file of settings.')
parser.add_argument('--n_iters', type=int, default=1000, help='Json file of settings.')


# def set_device(args):
#     torch.cuda.set_device(args.gpuid)


import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

# PyTorch convolutional layers require 4-dimensional inputs, in NCHW order
corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
               "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
               "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]


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



def get_x_embedded(agent, loader, device, path, dataset_name):
    x_embedded, y_embedded = agent._x_embedding(model=agent._network,
                                                loader=loader, device=device)
    torch.save(x_embedded, "{}/X_{}.pt".format(path, dataset_name))
    torch.save(y_embedded, "{}/Y_{}.pt".format(path, dataset_name))

    # x_embedded = torch.load("{}/X_{}.pt".format(path, dataset_name)).to(device)
    # y_embedded = torch.load("{}/Y_{}.pt".format(path, dataset_name)).to(device)
    print("Saving at {}".format("{}/X_{}.pt".format(path, dataset_name)))
    return x_embedded, y_embedded


def get_cifar10_c_datasets_embedded(agent, c_datasets, path, dataset_name, device):
    print("===>get_cifar10_c_datasets_embedded()<===")
    xs, ys = None, None
    for i, (
            (x_old, y_old), (x_1_old, y_1_old), (x_2_old, y_2_old), (x_3_old, y_3_old), (x_4_old, y_4_old),
            (x_5_old, y_5_old),
            (x_new, y_new), (x_1_new, y_1_new), (x_2_new, y_2_new), (x_3_new, y_3_new), (x_4_new, y_4_new),
            (x_5_new, y_5_new)) in enumerate(zip(
        c_datasets["0"], c_datasets["1"], c_datasets["2"], c_datasets["3"], c_datasets["4"], c_datasets["5"],
        c_datasets["50"], c_datasets["51"], c_datasets["52"], c_datasets["53"], c_datasets["54"], c_datasets["55"],

    )):

        if xs is None:
            # xs = x_old
            xs, x_1_old, x_2_old, x_3_old, x_4_old, x_5_old = x_old.to(device), x_1_old.to(device), x_2_old.to(
                device), x_3_old.to(device), x_4_old.to(
                device), x_5_old.to(device)
            xs = torch.concat([xs, x_1_old, x_2_old, x_3_old, x_4_old, x_5_old], dim=0).to(device)

            ys = y_old
            ys, y_1_old, y_2_old, y_3_old, y_4_old, y_5_old = ys.to(device), y_1_old.to(device), y_2_old.to(
                device), y_3_old.to(
                device), y_4_old.to(device), y_5_old.to(device)
            ys = torch.concat(
                [ys, (torch.zeros(y_1_old.shape) + 11).to(device), (torch.zeros(y_2_old.shape) + 12).to(device),
                 (torch.zeros(y_3_old.shape) + 13).to(device), (torch.zeros(y_4_old.shape) + 14).to(device),
                 (torch.zeros(y_5_old.shape) + 15).to(device)])

            x_new, x_1_new, x_2_new, x_3_new, x_4_new, x_5_new = x_new.to(device), x_1_new.to(device), x_2_new.to(
                device), x_3_new.to(device), x_4_new.to(
                device), x_5_new.to(device)
            xs = torch.concat([xs, x_new, x_1_new, x_2_new, x_3_new, x_4_new, x_5_new], dim=0).to(device)

            y_new, y_1_new, y_2_new, y_3_new, y_4_new, y_5_new = y_new.to(device), y_1_new.to(device), y_2_new.to(
                device), y_3_new.to(
                device), y_4_new.to(device), y_5_new.to(device)
            ys = torch.concat(
                [ys, (torch.zeros(y_new.shape) + 50).to(device), (torch.zeros(y_1_new.shape) + 51).to(device),
                 (torch.zeros(y_2_new.shape) + 52).to(device), (torch.zeros(y_3_new.shape) + 53).to(device),
                 (torch.zeros(y_4_new.shape) + 54).to(device), (torch.zeros(y_5_new.shape) + 55).to(device)])

        else:
            xs, x_old, x_1_old, x_2_old, x_3_old, x_4_old, x_5_old = xs.to(device), x_old.to(device), x_1_old.to(
                device), x_2_old.to(
                device), x_3_old.to(device), x_4_old.to(device), x_5_old.to(device)
            ys, y_old, y_1_old, y_2_old, y_3_old, y_4_old, y_5_old = ys.to(device), y_old.to(device), y_1_old.to(
                device), y_2_old.to(
                device), y_3_old.to(
                device), y_4_old.to(device), y_5_old.to(device)
            xs = torch.concat([xs, x_old, x_1_old, x_2_old, x_3_old, x_4_old, x_5_old], dim=0)
            ys = torch.concat(
                [ys, (torch.zeros(y_old.shape)).to(device), (torch.zeros(y_1_old.shape) + 11).to(device),
                 (torch.zeros(y_2_old.shape) + 12).to(device),
                 (torch.zeros(y_3_old.shape) + 13).to(device), (torch.zeros(y_4_old.shape) + 14).to(device),
                 (torch.zeros(y_5_old.shape) + 15).to(device)])

            x_new, x_1_new, x_2_new, x_3_new, x_4_new, x_5_new = x_new.to(device), x_1_new.to(device), x_2_new.to(
                device), x_3_new.to(device), x_4_new.to(
                device), x_5_new.to(device)
            y_new, y_1_new, y_2_new, y_3_new, y_4_new, y_5_new = y_new.to(device), y_1_new.to(device), y_2_new.to(
                device), y_3_new.to(device), y_4_new.to(device), y_5_new.to(device)

            xs = torch.concat([xs, x_new, x_1_new, x_2_new, x_3_new, x_4_new, x_5_new], dim=0)
            ys = torch.concat([ys, (torch.zeros(y_new.shape) + 50), (torch.zeros(y_1_new.shape) + 51),
                               (torch.zeros(y_2_new.shape) + 52).to(device),
                               (torch.zeros(y_3_new.shape) + 53).to(device),
                               (torch.zeros(y_4_new.shape) + 54).to(device),
                               (torch.zeros(y_5_new.shape) + 55).to(device)])

            # xs, x_1, x_2, x_3, x_4, x_5, x_6 = xs.to(device), x_1.to(device), x_2.to(device), x_3.to(device), x_4.to(
            #     device), x_5.to(device), x_6.to(device)

            # ys = torch.concat(
            #     [ys, y, (torch.zeros(y_1.shape) + 11).to(device), (torch.zeros(y_2.shape) + 12).to(device),
            #      (torch.zeros(y_3.shape) + 13).to(device), (torch.zeros(y_4.shape) + 14).to(device),
            #      (torch.zeros(y_5.shape) + 15).to(device), (torch.zeros(y_6.shape) + 50).to(device)])
    print("==> xs:", xs.shape, "\tys:", ys.shape)
    combine_dataset = TensorDataset(xs, ys)
    combine_loader = torch.utils.data.DataLoader(combine_dataset, batch_size=20480, shuffle=False)
    x_embedded, y_embedded = agent._x_embedding(model=agent._network,
                                                loader=combine_loader, device=device)

    torch.save(x_embedded, "{}/X_{}.pt".format(path, dataset_name))
    torch.save(y_embedded, "{}/Y_{}.pt".format(path, dataset_name))

    # x_embedded = torch.load("{}/X_{}".format(path, dataset_name)).to(device)
    # y_embedded = torch.load("{}/Y_{}".format(path, dataset_name)).to(device)
    print("Saving at {}".format("{}/X_{}.pt".format(path, dataset_name)))
    return x_embedded, y_embedded


def select_class_examples(dataset, target_labels=[0, 1]):
    # Create a list of indices corresponding to the target label
    target_indices = [i for i, (_, label) in enumerate(dataset) if label in target_labels]
    # Create a subset of the CIFAR10 dataset containing only the target images
    target_dataset = torch.utils.data.Subset(dataset, target_indices)
    return target_dataset

def get_transforms(pycil_args):
    # _agent._network.cuda()
    # print("_args:", _args)
    if ("not" in pycil_args["normalize"]):
        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        print("NOT NORMALIZED INPUT...")
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor()#,
            # transforms.Normalize(
            #     mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            # )
        ])
        print("NORMALIZED INPUT...")
    return val_transform


def valid_cifar10(task, agent, val_transform, target_labels, args):
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                               transform=val_transform)
    val_dataset = select_class_examples(dataset=val_dataset, target_labels=target_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20480, shuffle=False)
    val_acc = agent._cl_compute_accuracy(model=agent._network, loader=val_loader, device=args.gpuid)
    print("PYCIL ACC at Task{}=".format(task), val_acc)
    return val_acc

#
# def tsne_cifar10(path, agent, val_transform, args, dataset_name="cifar10", class_to_viz=[0, 1]):
#     # class_to_viz =
#     # for tg_cls in class_to_viz:
#
#     # c_ys = None
#     # for c in corruptions:
#     #     r_acc = []
#     #     r_acc.append(val_acc)
#     #     c_datasets = {}
#     cifar10_classes = select_class_examples(
#         torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
#                                      transform=val_transform),
#         target_labels=class_to_viz)
#     cifar10_loader = DataLoader(cifar10_classes, batch_size=10000)
#
#     x_embedded, y_embedded = agent._x_embedding(model=agent._network,
#                                                 loader=cifar10_loader, device=args.gpuid)
#
#     torch.save(x_embedded, "{}/X_{}_[{}].pt".format(path, dataset_name, "_".join(map(str, class_to_viz))))
#     torch.save(y_embedded, "{}/Y_{}_[{}].pt".format(path, dataset_name, "_".join(map(str, class_to_viz))))
#
#     (t_x_embed, t_y_embed) = get_tsne_embedded(x_embedded, y_embedded, n_iter=args.n_iters,
#                                                perplexity=args.perplexity, num_neighbors=args.num_neighbors,
#                                                dataset_name=dataset_name, path=path)
#     del x_embedded
#     del y_embedded
#     tsne_c_cifar10_viz(X_Tsne=t_x_embed, Y_Tsne=t_y_embed, dataset_name=dataset_name, path=path, title=dataset_name)
#     del t_x_embed
#     del t_y_embed
#
#     # return val_acc


def tsne_c_cifar10_viz(X_Tsne, Y_Tsne, path, dataset_name, title):
    import numpy as np
    from tsnecuda import TSNE
    import matplotlib.pyplot as plt
    # Create the figure
    plt.figure(figsize=(4, 4))
    MARKERS = ['o', 'o', 'o', 'o', 'o', 'o',
               '*', '*', '*', '*', '*', '*']

    CIFAR10_MARKERS =['o', 'o','o','o','o','o','o','o','o','o','o','o','o'] #'*' , 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
    colors = ["#1515ff", "#2a2aff", "#4040ff", "#5555ff", "#6a6aff", "#8080ff",
              "#ff1515", "#ff2a2a", "#ff4040", "#ff5555", "#ff6a6a", "#ff8080"]
    cifar10_colors = ['blue', 'green', 'red', 'cyan', '#FFD700', 'magenta', 'black', '#FFA500', 'purple', '#00CED1', '#8B008B']

    labels = torch.unique(Y_Tsne.cpu()).numpy()

    for i in range(len(labels)):
        #     if(i%2==0):
        print("===> {} Viz Lbl={} \t Count={} \t C-Count={}".format(labels[i], torch.unique(
            Y_Tsne[Y_Tsne.cpu() == labels[i]].cpu()), Y_Tsne.cpu().shape,
                                                                    X_Tsne[Y_Tsne.cpu() == labels[i]].shape))

        if title == "cifar10":
            plt.scatter(
                x=X_Tsne[Y_Tsne.cpu() == labels[i]][:, 0],
                y=X_Tsne[Y_Tsne.cpu() == labels[i]][:, 1],
                c=cifar10_colors[i],
                label=labels[i],
                marker=CIFAR10_MARKERS[i],
                #     cmap=plt.colormaps.get_cmap('Paired'),
                alpha=1,
                s=1)
        else:
            plt.scatter(
                x=X_Tsne[Y_Tsne.cpu() == labels[i]][:, 0],
                y=X_Tsne[Y_Tsne.cpu() == labels[i]][:, 1],
                c=colors[i],
                label=labels[i],
                marker=MARKERS[i],
                #     cmap=plt.colormaps.get_cmap('Paired'),
                alpha=1,
                s=1)
    print(labels)
    plt.legend(fontsize='large')
    legend = plt.legend()
    plt.title(title)
    for handle in legend.legendHandles:
        handle.set_sizes([30])  # Set the marker size here
    # plt.savefig("{}/tsne_{}.png".format(path, dataset_name))
    # plt.show()
    plt.savefig("{}/tsne_{}.png".format(path, dataset_name))
    plt.show()

def get_tsne_embedded(x_embedded, y_embedded, path, dataset_name, n_iter=1, perplexity=40, num_neighbors=128):
    print("X_Embedded: {} \t Y_Embedded:{}".format(x_embedded.shape, y_embedded.shape))
    tsne_x_embedded = TSNE(n_iter=n_iter, verbose=1, perplexity=perplexity, num_neighbors=num_neighbors).fit_transform(
        x_embedded.cpu())
    torch.save(x_embedded, "{}/X_sne_{}.pt".format(path, dataset_name))
    torch.save(y_embedded, "{}/Y_sne_{}.pt".format(path, dataset_name))
    print("Saving at {}".format("{}/X_sne_{}".format(path, dataset_name)))
    return tsne_x_embedded, y_embedded

def tsne_cifar10(path, agent, val_transform, args, dataset_name="cifar10", class_to_viz=[0, 1]):
    # class_to_viz =
    # for tg_cls in class_to_viz:

    # c_ys = None
    # for c in corruptions:
    #     r_acc = []
    #     r_acc.append(val_acc)
    #     c_datasets = {}
    cifar10_classes = select_class_examples(
        torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                     transform=val_transform),
        target_labels=class_to_viz)
    cifar10_loader = DataLoader(cifar10_classes, batch_size=10000)

    x_embedded, y_embedded = agent._x_embedding(model=agent._network,
                                                loader=cifar10_loader, device=args.gpuid)

    torch.save(x_embedded, "{}/X_{}_[{}].pt".format(path, dataset_name, "_".join(map(str, class_to_viz))))
    torch.save(y_embedded, "{}/Y_{}_[{}].pt".format(path, dataset_name, "_".join(map(str, class_to_viz))))

    (t_x_embed, t_y_embed) = get_tsne_embedded(x_embedded, y_embedded, n_iter=args.n_iters,
                                               perplexity=args.perplexity, num_neighbors=args.num_neighbors,
                                               dataset_name=dataset_name, path=path)
    del x_embedded
    del y_embedded
    tsne_c_cifar10_viz(X_Tsne=t_x_embed, Y_Tsne=t_y_embed, dataset_name=dataset_name, path=path, title=dataset_name)
    del t_x_embed
    del t_y_embed

    # return val_acc


def tsne_adv_cifar10(task, path, agent, val_transform, args, dataset_name="cifar10", old_class=[0], new_class=[1]):
    ATTACKS = [
        {"fgsm": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192, 224]},
        {"pgd_linf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128, 160, 192, 224]},
        {"pgd_l2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.0, 12.,
                    14., 16.]}
    ]
    FGSM_SELECTED = [2, 5, 9, 12, 15]
    PGD_L2_SELECTED = [2, 5, 9, 12, 15]
    PGD_LINF_SELECTED = [2, 4, 6, 8, 9]
    for ATCK in ATTACKS:
        c_datasets = {}
        cifar10_classes = select_class_examples(
            torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                         transform=val_transform), target_labels=old_class)
        c_datasets["0"] = DataLoader(cifar10_classes, batch_size=20480)

        cifar10_classes = select_class_examples(
            torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform),
            target_labels=new_class)
        new_class_index = 50
        c_datasets[str(new_class_index)] = DataLoader(cifar10_classes, batch_size=20480)

        for attack, epss in ATCK.items():

            for i, eps in enumerate(epss):
                print("I:", i)
                if i >= 5:  # Load only first n adversarial CIFAR10 datsets.
                    break
                if(attack == "fgsm"):
                    atk_indx = FGSM_SELECTED[i]
                elif(attack =="pgd_linf"):
                    atk_indx = PGD_LINF_SELECTED[i]
                elif(attack =="pgd_l2"):
                    atk_indx = PGD_L2_SELECTED[i]
                else:
                    print("UNKNOWN ATTACK")

                x, y = get_adv_dataset(eps=epss[atk_indx], attack=attack)
                adv_dataset = CCIFARTensorDataset(x, y, transform=val_transform)

                #########
                old_classes = select_class_examples(adv_dataset, target_labels=old_class)
                c_datasets[str(i + 1)] = DataLoader(old_classes, batch_size=20480)

                adv_dataset = CCIFARTensorDataset(x, y, transform=val_transform)
                new_classes = select_class_examples(adv_dataset, target_labels=new_class)
                c_datasets[str(new_class_index + (i + 1))] = DataLoader(new_classes, batch_size=20480)
                #########

            ####
            t_path = "{}/T_{}_old_{}_new_{}_Adv_CIFAR10".format(path, str(task), old_class[0], new_class[0])
            if not os.path.exists(t_path):
                os.mkdir(t_path)
            x_embed, y_embed = get_cifar10_c_datasets_embedded(model=agent._network, c_datasets=c_datasets,
                                                               path=t_path,
                                                               dataset_name="T_{}_{}_old_{}_new_{}_{}".format(str(task),
                                                                                                              dataset_name,
                                                                                                              old_class[
                                                                                                                  0],
                                                                                                              new_class[
                                                                                                                  0],
                                                                                                              attack),
                                                               device=args.gpuid)
            (t_x_embed, t_y_embed) = get_tsne_embedded(x_embed, y_embed, n_iter=args.n_iters,
                                                       perplexity=args.perplexity, num_neighbors=args.num_neighbors,
                                                       dataset_name="T_{}_{}_old_{}_new_{}_{}".format(str(task),
                                                                                                      dataset_name,
                                                                                                      old_class[0],
                                                                                                      new_class[0],
                                                                                                      attack),
                                                       path=t_path)
            tsne_c_cifar10_viz(X_Tsne=t_x_embed, Y_Tsne=t_y_embed,
                               dataset_name="T_{}_{}_old_{}_new_{}_{}".format(str(task), dataset_name, old_class[0],
                                                                              new_class[0], attack),
                               path=t_path, title=attack)

            #     robust_acc[c] = r_acc
            print("===" * 50)
        #     break
        # break


def tsne_cifar10_c(task, path, agent, val_transform, args, dataset_name="cifar10", old_class=[0], new_class=[1]):
    CIFAR10_C_Path = "/home/khanhi83/ACLWorkspace/CIFAR-10-C"
    # robust_acc = {}
    # class_to_viz = [0, 1]
    # for tg_cls in class_to_viz:

    for c in corruptions:
        print("===" * 50)
        print("{} of CIFAR10-C".format(c))
        # r_acc = []
        # r_acc.append(val_acc)
        c_datasets = {}
        cifar10_classes = select_class_examples(
            torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                         transform=val_transform), target_labels=old_class)
        c_datasets["0"] = DataLoader(cifar10_classes, batch_size=20480)

        cifar10_classes = select_class_examples(
            torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform),
            target_labels=new_class)
        new_class_index = 50
        c_datasets[str(new_class_index)] = DataLoader(cifar10_classes, batch_size=20480)

        for s in range(1, 6):
            # c_loader = DataLoader(load_cifar10(path=CIFAR10_C_Path, corruptions=c, severity=s), batch_size=1024)
            # print(len(subset))
            # t_acc = agent._adv_compu
            # te_accuracy(model=agent._network, loader=c_loader, device=args.gpuid)
            print("{} with severity {}".format(c, s))
            # r_acc.append(t_acc)
            # x_embed, y_embed = get_x_embedded(model=agent._network, loader=c_loader,
            #                                   path=path, dataset_name="{}_{}_{}".format(pycil_args["dataset"], c, s), device=args.gpuid)
            # (t_x_embed, t_y_embed) = get_tsne_embedded(x_embed, y_embed, n_iter=args.n_iters,
            #                                    perplexity=args.perplexity, num_neighbors=args.num_neighbors,
            #                                            dataset_name="{}_{}_{}".format(pycil_args["dataset"], c, s), path=path)
            # tsne_viz(X_Tsne=t_x_embed, Y_Tsne=t_y_embed, dataset_name="{}_{}_{}".format(pycil_args["dataset"], c, s), path=path)

            old_classes = select_class_examples(load_cifar10(path=CIFAR10_C_Path, corruptions=c, severity=s),
                                                target_labels=old_class)
            c_datasets[str(s)] = DataLoader(old_classes, batch_size=20480)

            new_classes = select_class_examples(load_cifar10(path=CIFAR10_C_Path, corruptions=c, severity=s),
                                                target_labels=new_class)
            c_datasets[str(new_class_index + s)] = DataLoader(new_classes, batch_size=20480)

        # model, c_datasets, path, dataset_name, device
        # model, c_datasets, path, dataset_name, device

        t_path = "{}/T_{}_old_{}_new_{}_CIFAR10_C".format(path, str(task), old_class[0], new_class[0])
        if not os.path.exists(t_path):
            os.mkdir(t_path)

        # TODO Need to Add below and replace the dataset_name with it.
        t_dataset_name = "{}_{}".format(dataset_name, c)
        x_embed, y_embed = get_cifar10_c_datasets_embedded(model=agent._network, c_datasets=c_datasets,
                                                           path=t_path,
                                                           dataset_name="T_{}_{}_old_{}_new_{}".format(str(task),
                                                                                                       t_dataset_name,
                                                                                                       old_class[0],
                                                                                                       new_class[0]),
                                                           device=args.gpuid)
        (t_x_embed, t_y_embed) = get_tsne_embedded(x_embed, y_embed, n_iter=args.n_iters,
                                                   perplexity=args.perplexity, num_neighbors=args.num_neighbors,
                                                   dataset_name="T_{}_{}_old_{}_new_{}".format(str(task), t_dataset_name,
                                                                                               old_class[0],
                                                                                               new_class[0]),
                                                   path=t_path)
        tsne_c_cifar10_viz(X_Tsne=t_x_embed, Y_Tsne=t_y_embed,
                           dataset_name="T_{}_{}_old_{}_new_{}_{}".format(str(task), t_dataset_name, old_class[0],
                                                                          new_class[0], c),
                           path=t_path, title=c)

        #     robust_acc[c] = r_acc
        print("===" * 50)
        # break
    # from pprint import pprint
    #
    # pprint(robust_acc)
    # torch.save(robust_acc, "{}/{}_cifar10c_acc.pt".format(path, pycil_args["model_name"]))



def perform_tsne_on_model(task_id, agent, args, pycil_args, class_to_viz=[0, 1]):
    print("pycil_args=", pycil_args)
    path = "{}/T_SNE_{}".format(args.tsne_path, pycil_args["dataset"].upper())
    if not os.path.exists(path):
        os.mkdir(path)

    path = "{}/{}_{}_{}_{}_{}".format(path, pycil_args["model_name"], pycil_args["use_adv_weight"],
                                      pycil_args["memory_size"],
                                      pycil_args["init_cls"], pycil_args["increment"])
    if not os.path.exists(path):
        os.mkdir(path)
    print("PATH={}".format(path))
    val_transform = get_transforms(pycil_args)

    t_path = "{}/T_{}_class_[{}]".format(path, task_id, "_".join(map(str, class_to_viz)))
    if not os.path.exists(t_path):
        os.mkdir(t_path)
    print("T_PATH={}".format(t_path))
    tsne_cifar10(path=t_path, agent=agent, val_transform=val_transform,
                 args=args, dataset_name="cifar10", class_to_viz=class_to_viz)

    tsne_cifar10_c(task=task_id, path=t_path, agent=agent, val_transform=val_transform, args=args,
                   dataset_name="cifar10", old_class=[class_to_viz[0]],
                   new_class=[class_to_viz[1]])

    tsne_adv_cifar10(task=task_id, path=t_path, agent=agent, val_transform=val_transform, args=args,
                     dataset_name="cifar10", old_class=[class_to_viz[0]],
                     new_class=[class_to_viz[1]])