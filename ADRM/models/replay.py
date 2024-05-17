import logging
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8


init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 70
lrate = 0.1
milestones = [30, 50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2


class Replay(BaseLearner):
    def __init__(self, args):

        global init_epoch, init_lr, init_lr_decay, init_weight_decay, epochs, lrate, lrate_decay, weight_decay, T, batch_size
        init_epoch = args["replay_init_epoch"]
        init_lr = args["replay_init_lr"]
        init_lr_decay = args["replay_init_lr_decay"]
        init_weight_decay = args["replay_init_weight_decay"]
        epochs = args["replay_epochs"]
        lrate = args["replay_lrate"]
        batch_size = args["replay_batch_size"]
        lrate_decay = args["replay_lrate_decay"]
        weight_decay = args["replay_weight_decay"]
        T = args["replay_T"]
        super().__init__(args)
        self._network = IncrementalNet(args, False)


    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )


        # Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train",  mode="train",
            appendent=self._get_memory(),
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

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
    def get_adv_memory(self, trsfm):
        (combined_succeed_adv_attacked_imgs, combined_succeed_adv_attacked_imgs_lbls), (
        combined_not_succeed_adv_attacked_imgs, combined_not_succeed_adv_attack_lbls) = (None, None), (None, None)
        if self._get_memory() is None:
            logging.info("==> Learning task # {} with memory buffer size={}".format(0, 0))
        else:
            # logging.info("==> Learning task # {} with memory buffer size={}".format(task, len(model._get_memory()[1])))
            data, targets = self._get_memory()
            b_data, b_targets = [], []
            b_adv_data, b_adv_targets = [], []
            # print("Targets:", type(targets), " ", targets.shape, " Unique:", np.unique(targets))
            for i in np.unique(targets):
                indices = np.where(targets == i)
                # print(
                #     "==> Class {} has {} samples available.".format(np.unique(targets[indices]), len(targets[indices])))

                trsf = transforms.Compose([[
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=63 / 255),
                    transforms.ToTensor()
                ]])
                # print("indices=", type(indices), "  ", indices, " Data Shape:", data[indices].shape)

                # data = [torch.from_numpy(x) for x in data[indices]]
                # x.transpose(1, 2, 0)
                image = [trsfm(Image.fromarray(x)) for x in data[indices]]
                image = torch.stack(image)
                # print("-->IMAGEA:", type(image), " data:", type(data), " ", data.shape, " ", image.shape, " ", targets.shape, " Unique:", np.unique(targets))
                t_targets = torch.from_numpy(targets[indices]).clone().to(self._device)
                o_inputs, adv_inputs, success = self._get_adv_examples(self._network, inputs=image.clone().to(self._device),
                                                                targets=t_targets)

                # prd_labels = torch.argmax(self._network(adv_inputs), dim=1)
                # print("Success Ratio {}:".format(torch.sum(success) / len(t_targets)))
                # print(type(adv_inputs[0]), " ", len(adv_inputs[0]), ":", type(o_inputs[0]), len(o_inputs[0]), ":", type(success))

                # print("adv_inputs:", adv_inputs[0].shape, " o_inputs:", o_inputs[0].shape, " success:", type(success), " ", len(success[0]))
                succeed_adv_attacked_imgs = adv_inputs[0][success[0]]
                succeed_adv_attacked_imgs_lbls = torch.from_numpy(targets[indices])[success[0]] # Added later
                not_succeed_adv_attacked_imgs = o_inputs[0][~success[0]]
                not_succeed_adv_attack_lbls = torch.from_numpy(targets[indices])[~success[0]]
                # trsf(not_succeed_adv_attacked_imgs)
                # trsf(succeed_adv_attacked_imgs)
                b_data.append(not_succeed_adv_attacked_imgs)
                b_targets.append(not_succeed_adv_attack_lbls)

                # print(type(succeed_adv_attacked_imgs), "\t", type(succeed_adv_attacked_imgs_lbls),
                #       "\t", type(not_succeed_adv_attacked_imgs), "\t", type(not_succeed_adv_attack_lbls))


                # print("Success Ratio on class {}:".format(torch.unique(t_targets)), torch.sum(success) / len(t_targets))
                # print("X_Adv:", len(succeed_adv_attacked_imgs), " X_Adv_Lbls:", len(succeed_adv_attacked_imgs_lbls),
                #       "X_Original:", len(not_succeed_adv_attacked_imgs), "X_Original_Lbls:",len(not_succeed_adv_attack_lbls),
                #       "=", (len(succeed_adv_attacked_imgs) + len(not_succeed_adv_attacked_imgs)))

                # if (combined_succeed_adv_attacked_imgs is None):
                # else:
                #
                # if (combined_succeed_adv_attacked_imgs is None):
                #     else:

                if (combined_not_succeed_adv_attacked_imgs is None):
                    combined_not_succeed_adv_attacked_imgs = not_succeed_adv_attacked_imgs
                    combined_not_succeed_adv_attack_lbls = not_succeed_adv_attack_lbls
                else:
                    combined_not_succeed_adv_attacked_imgs = torch.concat(
                        [combined_not_succeed_adv_attacked_imgs, not_succeed_adv_attacked_imgs], dim=0)
                    combined_not_succeed_adv_attack_lbls = torch.concat(
                        [combined_not_succeed_adv_attack_lbls, not_succeed_adv_attack_lbls])

                if (combined_succeed_adv_attacked_imgs is None):
                    combined_succeed_adv_attacked_imgs = succeed_adv_attacked_imgs
                    combined_succeed_adv_attacked_imgs_lbls = succeed_adv_attacked_imgs_lbls
                else:
                    combined_succeed_adv_attacked_imgs = torch.concat(
                        [combined_succeed_adv_attacked_imgs, succeed_adv_attacked_imgs], dim=0)
                    combined_succeed_adv_attacked_imgs_lbls = torch.concat(
                        [combined_succeed_adv_attacked_imgs_lbls, succeed_adv_attacked_imgs_lbls], dim=0)


        # return torch.from_numpy(np.concatenate(not_succeed_adv_attacked_imgs)), torch.from_numpy(np.concatenate(not_succeed_adv_attack_lbls))
        return  (combined_succeed_adv_attacked_imgs.clone().to(self.args["device"][0]),
                 combined_succeed_adv_attacked_imgs_lbls.clone().to(self.args["device"][0])), \
                (combined_not_succeed_adv_attacked_imgs.clone().to(self.args["device"][0]),
                 combined_not_succeed_adv_attack_lbls.clone().to(self.args["device"][0]))

                # visualize(data[indices][:64], targets[indices][:64], str(task), row=8, cols=8)
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logging.info(
                #     "==> Init:Input ({}, {}) # {} {}".format(inputs[0].min().item(), inputs[0].max().item(), inputs.shape,
                #                                         targets.shape))

                if self.args["adv_train"]:
                    # logging.info("==> Adversarial training...")
                    for i in range(self.args["adv_train_steps"]):
                        _, adv_inputs, success = self._get_adv_examples(self._network, inputs=inputs.clone(), targets=targets.clone())
                        logits = self._network(torch.concat([inputs, adv_inputs[0]]))["logits"]
                        loss = F.cross_entropy(logits, torch.concat([targets, targets.clone()]))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print("Adv_Loss:", loss.item())

                        # logits = self._network(inputs)["logits"]
                        # loss = F.cross_entropy(logits, targets)
                        # optimizer.zero_grad()
                        # loss.backward()
                        # optimizer.step()
                        losses += loss.item()
                        _, preds = torch.max(logits, dim=1)
                        correct += preds.eq(torch.concat([targets, targets.clone()]).expand_as(preds)).cpu().sum()
                        total += len(torch.concat([targets, targets.clone()]))
                else:
                    # logging.info("==> Standard training...")
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                if(self.args["adversarial_robust_reply"]):
                    (combined_succeed_adv_attacked_imgs, combined_succeed_adv_attacked_imgs_lbls), (combined_not_succeed_adv_attacked_imgs,
                                                                                                    combined_not_succeed_adv_attack_lbls) = self.get_adv_memory(trsfm=transforms.Compose([*self.data_manager._train_trsf, *self.data_manager._common_trsf]))
                    # logging.info("==> Adv (succeed) memory count {} classes {} U-Classes {}".format(len(combined_succeed_adv_attacked_imgs), len(combined_succeed_adv_attacked_imgs_lbls), torch.unique(combined_succeed_adv_attacked_imgs_lbls)))
                    # logging.info("==> Adv (not-succeed) memory count {} classes {} U-Classes {}".format(len(combined_not_succeed_adv_attacked_imgs),
                    #                                                          len(combined_not_succeed_adv_attack_lbls), torch.unique(combined_not_succeed_adv_attack_lbls)))

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logits = self._network(inputs)["logits"]
                #
                # loss_clf = F.cross_entropy(logits, targets)
                # loss = loss_clf
                #
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # losses += loss.item()
                #
                # # acc
                # _, preds = torch.max(logits, dim=1)
                # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                # total += len(targets)

                if self.args["adv_train"]:

                    for i in range(self.args["adv_train_steps"]):#self._adv_train_steps):
                        # logging.info("==> Adversarial training {}...".format(i))
                        # print("==> Adv_step taken...")
                        _, adv_inputs, success = self._get_adv_examples(self._network, inputs=inputs.clone(), targets=targets.clone())
                        # adv_inputs = adv_inputs#.to("cuda:5")
                        # print("X:", inputs.shape, " Y:", targets.shape)
                        # print("Adding Adv: X:", adv_inputs[0].shape, " Y:", len(targets))
                        # inputs = torch.cat([inputs, adv_inputs[0]])
                        # targets= torch.cat([targets, targets.clone()])
                        # print("Added: X:", inputs.shape, " Y:", len(targets))

                        if self.args["adversarial_robust_reply"] == 1:
                            x_to_net = torch.concat([inputs, combined_not_succeed_adv_attacked_imgs, adv_inputs[0]])
                            y_to_net = torch.concat([targets, combined_not_succeed_adv_attack_lbls, targets.clone()])
                        elif self.args["adversarial_robust_reply"] == 2:
                            x_to_net = torch.concat([inputs, combined_not_succeed_adv_attacked_imgs, adv_inputs[0], combined_succeed_adv_attacked_imgs])
                            y_to_net = torch.concat([targets, combined_not_succeed_adv_attack_lbls, targets.clone(), combined_succeed_adv_attacked_imgs_lbls])
                        else:
                            x_to_net = torch.concat([inputs, adv_inputs[0]])
                            y_to_net = torch.concat([targets, targets.clone()])
                        combined_not_succeed_adv_attacked_imgs
                        logits = self._network(x_to_net)["logits"]
                        loss = F.cross_entropy(logits, y_to_net)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # print("Adv_Loss:", loss.item())

                        # logits = self._network(inputs)["logits"]
                        # loss = F.cross_entropy(logits, targets)
                        # optimizer.zero_grad()
                        # loss.backward()
                        # optimizer.step()
                        losses += loss.item()
                        _, preds = torch.max(logits, dim=1)
                        # correct += preds.eq(torch.cat([targets, targets.clone()]).expand_as(preds)).cpu().sum()
                        correct += preds.eq(y_to_net.expand_as(preds)).cpu().sum()

                        # total += len(torch.cat([targets, targets.clone()]))
                        total += len(y_to_net)
                else:

                    # logging.info("==> Standard training...")

                    x_to_net = inputs
                    y_to_net = targets

                    if self.args["adversarial_robust_reply"] == 1:
                        # print("==> adversarial_robust_reply={}".format(self.args["adversarial_robust_reply"]))
                        # print("{}, {}, {}, {}".format(x_to_net.device, y_to_net.device, combined_not_succeed_adv_attacked_imgs.device,
                        #                               combined_not_succeed_adv_attack_lbls.device))
                        x_to_net = torch.concat([x_to_net.clone(), combined_not_succeed_adv_attacked_imgs])
                        y_to_net = torch.concat([y_to_net.clone(), combined_not_succeed_adv_attack_lbls])
                    elif self.args["adversarial_robust_reply"] == 2:
                        x_to_net = torch.concat([x_to_net.clone(), combined_not_succeed_adv_attacked_imgs, combined_succeed_adv_attacked_imgs])
                        y_to_net = torch.concat([y_to_net.clone(), combined_not_succeed_adv_attack_lbls, combined_succeed_adv_attacked_imgs_lbls])
                    else:
                        pass

                    logits = self._network(x_to_net)["logits"]

                    loss_clf = F.cross_entropy(logits, y_to_net)
                    loss = loss_clf

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    # acc
                    _, preds = torch.max(logits, dim=1)
                    # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    # total += len(targets)
                    correct += preds.eq(y_to_net.expand_as(preds)).cpu().sum()
                    total += len(y_to_net)

                    # Remove below
                    # logging.info("==> Standard training...")
                    # logits = self._network(inputs)["logits"]
                    # loss = F.cross_entropy(logits, targets)
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    # losses += loss.item()
                    #
                    # _, preds = torch.max(logits, dim=1)
                    # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    # total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

# import logging
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm
# import torch
# from torch import nn
# from torch import optim
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from models.base import BaseLearner
# from utils.inc_net import IncrementalNet
# from utils.toolkit import target2onehot, tensor2numpy
#
# EPSILON = 1e-8
#
#
# init_epoch = 200
# init_lr = 0.1
# init_milestones = [60, 120, 170]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005
#
#
# epochs = 70
# lrate = 0.1
# milestones = [30, 50]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 4
# T = 2
#
#
# class Replay(BaseLearner):
#     def __init__(self, args):
#         global init_epoch, init_lr, init_lr_decay, init_weight_decay, epochs, lrate, lrate_decay, weight_decay, T, batch_size
#         init_epoch = args["replay_init_epoch"]
#         init_lr = args["replay_init_lr"]
#         init_lr_decay = args["replay_init_lr_decay"]
#         init_weight_decay = args["replay_init_weight_decay"]
#         epochs = args["replay_epochs"]
#         lrate = args["replay_lrate"]
#         batch_size = args["replay_batch_size"]
#         lrate_decay = args["replay_lrate_decay"]
#         weight_decay = args["replay_weight_decay"]
#         T = args["replay_T"]
#         super().__init__(args)
#         self._network = IncrementalNet(args, False)
#
#
#     def after_task(self):
#         self._known_classes = self._total_classes
#         logging.info("Exemplar size: {}".format(self.exemplar_size))
#
#     def incremental_train(self, data_manager):
#         self.data_manager = data_manager
#         self._cur_task += 1
#         self._total_classes = self._known_classes + data_manager.get_task_size(
#             self._cur_task
#         )
#         self._network.update_fc(self._total_classes)
#         logging.info(
#             "Learning on {}-{}".format(self._known_classes, self._total_classes)
#         )
#
#
#         # Loader
#         train_dataset = data_manager.get_dataset(
#             np.arange(self._known_classes, self._total_classes), source="train",  mode="train",
#             appendent=self._get_memory(),
#         )
#
#         self.train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
#         )
#         test_dataset = data_manager.get_dataset(
#             np.arange(0, self._total_classes), source="test", mode="test"
#         )
#         self.test_loader = DataLoader(
#             test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
#         )
#
#         # Procedure
#         if len(self._multiple_gpus) > 1:
#             self._network = nn.DataParallel(self._network, self._multiple_gpus)
#         self._train(self.train_loader, self.test_loader)
#
#         self.build_rehearsal_memory(data_manager, self.samples_per_class)
#         if len(self._multiple_gpus) > 1:
#             self._network = self._network.module
#
#     def _train(self, train_loader, test_loader):
#         self._network.to(self._device)
#         if self._cur_task == 0:
#             optimizer = optim.SGD(
#                 self._network.parameters(),
#                 momentum=0.9,
#                 lr=init_lr,
#                 weight_decay=init_weight_decay,
#             )
#             scheduler = optim.lr_scheduler.MultiStepLR(
#                 optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
#             )
#             self._init_train(train_loader, test_loader, optimizer, scheduler)
#         else:
#             optimizer = optim.SGD(
#                 self._network.parameters(),
#                 lr=lrate,
#                 momentum=0.9,
#                 weight_decay=weight_decay,
#             )  # 1e-5
#             scheduler = optim.lr_scheduler.MultiStepLR(
#                 optimizer=optimizer, milestones=milestones, gamma=lrate_decay
#             )
#             self._update_representation(train_loader, test_loader, optimizer, scheduler)
#
#         # if model._get_memory() is None:
#         #     logging.info("==> Learning task # {} with memory buffer size={}".format(task, 0))
#         # else:
#         #     logging.info("==> Learning task # {} with memory buffer size={}".format(task, len(model._get_memory()[1])))
#         #     data, targets = model._get_memory()
#         #     print("Targets:", type(targets), " ", targets.shape, " Unique:", np.unique(targets))
#         #     for i in np.unique(targets):
#         #         indices = np.where(targets == i)
#         #         print(
#         #             "==> Class {} has {} samples available.".format(np.unique(targets[indices]), len(targets[indices])))
#         #
#         #         visualize(data[indices][:64], targets[indices][:64], str(task), row=8, cols=8)
#     def get_adv_memory(self, trsfm):
#
#         if self._get_memory() is None:
#             logging.info("==> Learning task # {} with memory buffer size={}".format(0, 0))
#         else:
#             # logging.info("==> Learning task # {} with memory buffer size={}".format(task, len(model._get_memory()[1])))
#             data, targets = self._get_memory()
#             b_data, b_targets = [], []
#             print("Targets:", type(targets), " ", targets.shape, " Unique:", np.unique(targets))
#             for i in np.unique(targets):
#                 indices = np.where(targets == i)
#                 print(
#                     "==> Class {} has {} samples available.".format(np.unique(targets[indices]), len(targets[indices])))
#
#                 trsf = transforms.Compose([[
#                     transforms.RandomCrop(32, padding=4),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.ColorJitter(brightness=63 / 255),
#                     transforms.ToTensor()
#                 ]])
#                 # print("indices=", type(indices), "  ", indices, " Data Shape:", data[indices].shape)
#
#                 # data = [torch.from_numpy(x) for x in data[indices]]
#                 # x.transpose(1, 2, 0)
#                 image = [trsfm(Image.fromarray(x)) for x in data[indices]]
#                 image = torch.stack(image)
#                 print("-->IMAGEA:", type(image), " data:", type(data), " ", data.shape, " ", image.shape, " ", targets.shape, " Unique:", np.unique(targets))
#                 t_targets = torch.from_numpy(targets[indices]).clone().to(self._device)
#                 o_inputs, adv_inputs, success = self._get_adv_examples(self._network, inputs=image.clone().to(self._device),
#                                                                 targets=t_targets)
#
#                 # prd_labels = torch.argmax(self._network(adv_inputs), dim=1)
#                 print("Success Ratio {}:".format(torch.sum(success) / len(t_targets)))
#                 # print(type(adv_inputs[0]), " ", len(adv_inputs[0]), ":", type(o_inputs[0]), len(o_inputs[0]), ":", type(success))
#
#                 print("adv_inputs:", adv_inputs[0].shape, " o_inputs:", o_inputs[0].shape, " success:", type(success), " ", len(success[0]))
#                 succeed_adv_attacked_imgs = adv_inputs[0][success[0]]
#                 not_succeed_adv_attacked_imgs = o_inputs[0][~success[0]]
#                 not_succeed_adv_attack_lbls = torch.from_numpy(targets[indices])[~success[0]]
#                 # trsf(not_succeed_adv_attacked_imgs)
#                 # trsf(succeed_adv_attacked_imgs)
#                 b_data.append(not_succeed_adv_attacked_imgs)
#                 b_targets.append(not_succeed_adv_attack_lbls)
#
#                 print("Success Ratio on class {}:".format(torch.unique(t_targets)), torch.sum(success) / len(t_targets))
#                 print("X_Adv:", len(succeed_adv_attacked_imgs), " X_Original:", len(not_succeed_adv_attacked_imgs), " =", (len(succeed_adv_attacked_imgs) + len(not_succeed_adv_attacked_imgs)))
#         # return torch.from_numpy(np.concatenate(not_succeed_adv_attacked_imgs)), torch.from_numpy(np.concatenate(not_succeed_adv_attack_lbls))
#         return  not_succeed_adv_attacked_imgs, not_succeed_adv_attack_lbls
#
#                 # visualize(data[indices][:64], targets[indices][:64], str(task), row=8, cols=8)
#     def _init_train(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(init_epoch))
#         for _, epoch in enumerate(prog_bar):
#             self._network.train()
#             losses = 0.0
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 # logging.info(
#                 #     "==> Init:Input ({}, {}) # {} {}".format(inputs[0].min().item(), inputs[0].max().item(), inputs.shape,
#                 #                                         targets.shape))
#
#                 if self.args["adv_train"]:
#                     # logging.info("==> Adversarial training...")
#                     for i in range(self.args["adv_train_steps"]):
#                         _, adv_inputs, success = self._get_adv_examples(self._network, inputs=inputs.clone(), targets=targets.clone())
#                         logits = self._network(torch.cat([inputs, adv_inputs[0]]))["logits"]
#                         loss = F.cross_entropy(logits, torch.cat([targets, targets.clone()]))
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()
#                         # print("Adv_Loss:", loss.item())
#
#                         # logits = self._network(inputs)["logits"]
#                         # loss = F.cross_entropy(logits, targets)
#                         # optimizer.zero_grad()
#                         # loss.backward()
#                         # optimizer.step()
#                         losses += loss.item()
#                         _, preds = torch.max(logits, dim=1)
#                         correct += preds.eq(torch.cat([targets, targets.clone()]).expand_as(preds)).cpu().sum()
#                         total += len(torch.cat([targets, targets.clone()]))
#                 else:
#                     # logging.info("==> Standard training...")
#                     logits = self._network(inputs)["logits"]
#                     loss = F.cross_entropy(logits, targets)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     losses += loss.item()
#
#                     _, preds = torch.max(logits, dim=1)
#                     correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                     total += len(targets)
#
#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
#
#             if epoch % 5 == 0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     init_epoch,
#                     losses / len(train_loader),
#                     train_acc,
#                     test_acc,
#                 )
#             else:
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     init_epoch,
#                     losses / len(train_loader),
#                     train_acc,
#                 )
#
#             prog_bar.set_description(info)
#
#         logging.info(info)
#
#     def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(epochs))
#         for _, epoch in enumerate(prog_bar):
#             self._network.train()
#             losses = 0.0
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 # b_data, b_lbl = self.get_adv_memory(trsfm=transforms.Compose([*self.data_manager._train_trsf, *self.data_manager._common_trsf]))
#                 # logging.info("==> Adv memory count {} classes {}".format(len(b_data), len(b_lbl)))
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 # logits = self._network(inputs)["logits"]
#                 #
#                 # loss_clf = F.cross_entropy(logits, targets)
#                 # loss = loss_clf
#                 #
#                 # optimizer.zero_grad()
#                 # loss.backward()
#                 # optimizer.step()
#                 # losses += loss.item()
#                 #
#                 # # acc
#                 # _, preds = torch.max(logits, dim=1)
#                 # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 # total += len(targets)
#
#                 if self.args["adv_train"]:
#
#                     for i in range(self.args["adv_train_steps"]):#self._adv_train_steps):
#                         # logging.info("==> Adversarial training {}...".format(i))
#                         # print("==> Adv_step taken...")
#                         _, adv_inputs, success = self._get_adv_examples(self._network, inputs=inputs.clone(), targets=targets.clone())
#                         # adv_inputs = adv_inputs#.to("cuda:5")
#                         # print("X:", inputs.shape, " Y:", targets.shape)
#                         # print("Adding Adv: X:", adv_inputs[0].shape, " Y:", len(targets))
#                         # inputs = torch.cat([inputs, adv_inputs[0]])
#                         # targets= torch.cat([targets, targets.clone()])
#                         # print("Added: X:", inputs.shape, " Y:", len(targets))
#                         logits = self._network(torch.cat([inputs, adv_inputs[0]]))["logits"]
#                         loss = F.cross_entropy(logits, torch.cat([targets, targets.clone()]))
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()
#                         # print("Adv_Loss:", loss.item())
#
#                         # logits = self._network(inputs)["logits"]
#                         # loss = F.cross_entropy(logits, targets)
#                         # optimizer.zero_grad()
#                         # loss.backward()
#                         # optimizer.step()
#                         losses += loss.item()
#                         _, preds = torch.max(logits, dim=1)
#                         correct += preds.eq(torch.cat([targets, targets.clone()]).expand_as(preds)).cpu().sum()
#                         total += len(torch.cat([targets, targets.clone()]))
#                 else:
#
#                     # logging.info("==> Standard training...")
#                     logits = self._network(inputs)["logits"]
#
#                     loss_clf = F.cross_entropy(logits, targets)
#                     loss = loss_clf
#
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     losses += loss.item()
#
#                     # acc
#                     _, preds = torch.max(logits, dim=1)
#                     correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                     total += len(targets)
#
#                     # Remove below
#                     # logging.info("==> Standard training...")
#                     # logits = self._network(inputs)["logits"]
#                     # loss = F.cross_entropy(logits, targets)
#                     # optimizer.zero_grad()
#                     # loss.backward()
#                     # optimizer.step()
#                     # losses += loss.item()
#                     #
#                     # _, preds = torch.max(logits, dim=1)
#                     # correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                     # total += len(targets)
#
#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
#             if epoch % 5 == 0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     epochs,
#                     losses / len(train_loader),
#                     train_acc,
#                     test_acc,
#                 )
#             else:
#                 info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
#                     self._cur_task,
#                     epoch + 1,
#                     epochs,
#                     losses / len(train_loader),
#                     train_acc,
#                 )
#             prog_bar.set_description(info)
#         logging.info(info)
