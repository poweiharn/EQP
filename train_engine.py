from models.net.vgg import vgg16_bn
from models.net.vgg import vgg16

from models.get_network import get_network

import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from util import WarmUpLR
#from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm






def train_engine(__C, train_loader, val_loader, test_loader):
    sim_data = {
        "test_acc": 0.0,
        "best_val_acc": 0.0,
        "best_val_epoch": 0,
        "gen_gap":0.0,
        "train_result": {}
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define dataloader

    train_length = len(train_loader.dataset)
    val_length = len(val_loader.dataset)
    test_length = len(test_loader.dataset)

    net = get_network(__C)
    #-------For CUDA---------
    if __C.cuda:
        net = net.cuda()

    #total_ops, total_params=profile(net,[128, 3, 32, 32])
    #print(total_ops)
    #print(total_params)

    # define optimizer and loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=__C.lr, momentum=0.9, weight_decay=5e-4)

    net.load_state_dict(torch.load("{}/init_nn_model.pt".format('ckpts/'+__C.dataset), map_location=device), strict=False)
    optimizer.load_state_dict(torch.load("{}/init_optimizer.pt".format('ckpts/'+__C.dataset), map_location=device))

    # define optimizer scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=__C.milestones, gamma=__C.lr_decay_rate)
    iter_per_epoch = len(train_loader)
    warmup_schedule = WarmUpLR(optimizer, iter_per_epoch * __C.warmup_epoch)



    '''
    # define tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(__C.tensorboard_log_dir,__C.name,__C.version))
    '''

    '''
    # define model save dir
    checkpoint_path = os.path.join(__C.ckpts_dir, __C.name, __C.version)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    
    '''

    # define the log save dir
    log_path = os.path.join(__C.result_log_dir, __C.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, __C.version + '.txt')

    # write the hyper parameters to log
    logfile = open(log_path, 'a+')
    logfile.write(str(__C))
    logfile.close()

    loss_sum = 0
    for epoch in range(1, __C.epoch):

        if epoch > __C.warmup_epoch:
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                train_scheduler.step(epoch)



        start = time.time()
        net.train()
        train_acc = 0.0
        train_correct = 0.0

        for step, (images, labels) in enumerate(tqdm(train_loader, colour="white", desc='[{Version}] [{Name}] Training Epoch: {epoch}'.format(Version=__C.version,Name=__C.name, epoch=epoch))):
            if epoch <= __C.warmup_epoch:
                with warnings.catch_warnings():  # Filter unnecessary warning.
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warmup_schedule.step()

            if __C.dataset == "PascalVOC":
                labels = labels.flatten()
            # -------For CUDA---------
            if __C.cuda:
                images = images.cuda()
                labels = labels.cuda()

            # using gradient accumulation step
            optimizer.zero_grad()
            loss_tmp = 0
            for accu_step in range(__C.gradient_accumulation_steps):
                loss_tmp = 0
                sub_images = images[accu_step * __C.sub_batch_size:
                                    (accu_step + 1) * __C.sub_batch_size]
                sub_labels = labels[accu_step * __C.sub_batch_size:
                                    (accu_step + 1) * __C.sub_batch_size]
                print(sub_images.size())
                outputs = net(sub_images)
                loss = loss_function(outputs, sub_labels)
                _, preds = outputs.max(1)
                train_correct += preds.eq(sub_labels).sum()
                loss.backward()
                # loss_tmp += loss.cpu().data.numpy() * __C.gradient_accumulation_steps
                # loss_sum += loss.cpu().data.numpy() * __C.gradient_accumulation_steps
                loss_tmp += loss.cpu().data.numpy()
                loss_sum += loss.cpu().data.numpy()

            optimizer.step()


            '''
            print(
                '[{Version}] [{Name}] Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss_tmp,
                    optimizer.param_groups[0]['lr'],
                    Version=__C.version,
                    Name=__C.name,
                    epoch=epoch,
                    trained_samples=step * __C.batch_size + len(images),
                    total_samples=len(train_loader.dataset)
                ))
            '''
        train_acc = train_correct.float() / train_length
        finish = time.time()
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

        if __C.eval_every_epoch:
            start = time.time()
            net.eval()
            val_loss = 0.0
            val_correct = 0.0
            for (images, labels) in val_loader:
                if __C.dataset == "PascalVOC":
                    labels = labels.flatten()
                # -------For CUDA---------
                if __C.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                eval_outputs = net(images)
                eval_loss = loss_function(eval_outputs, labels)
                val_loss += eval_loss.item()
                _, preds = eval_outputs.max(1)
                val_correct += preds.eq(labels).sum()
            finish = time.time()

            val_average_loss = val_loss / val_length  # 测试平均 loss
            val_acc = val_correct.float() / val_length # 测试准确率

            # save model after every "save_epoch" epoches and model with the best acc
            if sim_data["best_val_acc"] < val_acc.item():
                sim_data["best_val_acc"] = val_acc.item()
                sim_data["best_val_epoch"] = epoch



            # print the testing information
            #print('Evaluating Network.....')
            print('Validation set: Average loss: {:.4f}, Validation Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                val_average_loss,
                val_acc,
                finish - start
            ))
            print()

            sim_data["train_result"][str(epoch)] = {
                    "Train_Average_Loss": loss_sum/train_length,
                    "Train_Accuracy": train_acc.item(),
                    "LR": optimizer.param_groups[0]['lr'],
                    "Valid_Average_loss": val_average_loss,
                    "Valid_Accuracy":val_acc.item()
                }

            sim_data["gen_gap"] = abs(sim_data["train_result"][str(epoch)]["Train_Accuracy"] - sim_data["train_result"][str(epoch)]["Valid_Accuracy"])

            # update the result logfile
            logfile = open(log_path, 'a+')
            logfile.write(
                'Epoch: ' + str(epoch) +
                ', Train Average Loss: {:.4f}'.format(sim_data["train_result"][str(epoch)]["Train_Average_Loss"]) +
                ', Train Accuracy: {:.4f}'.format(sim_data["train_result"][str(epoch)]["Train_Accuracy"]) +
                ', Lr: {:.6f}'.format(sim_data["train_result"][str(epoch)]["LR"]) +
                ', Valid Average loss: {:.4f}'.format(sim_data["train_result"][str(epoch)]["Valid_Average_loss"]) +
                ', Validation Accuracy: {:.4f}'.format(val_acc) +
                '\n'
            )
            logfile.close()

            # update the tensorboard log file
            '''
            writer.add_scalar('[Epoch] Test/Average loss', test_average_loss, epoch)
            writer.add_scalar('[Epoch] Test/Accuracy', acc, epoch)
            '''

    test_loss = 0.0
    correct = 0.0
    start = time.time()
    for (images, labels) in test_loader:
        if __C.dataset == "PascalVOC":
            labels = labels.flatten()
        # -------For CUDA---------
        if __C.cuda:
            images = images.cuda()
            labels = labels.cuda()
        eval_outputs = net(images)
        eval_loss = loss_function(eval_outputs, labels)
        test_loss += eval_loss.item()
        _, preds = eval_outputs.max(1)
        correct += preds.eq(labels).sum()
    finish = time.time()

    test_average_loss = test_loss / len(test_loader.dataset)  # 测试平均 loss
    acc = correct.float() / len(test_loader.dataset)  # 测试准确率
    sim_data["test_acc"] = acc.item()
    print('Test set: Average loss: {:.4f}, Test Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_average_loss,
        acc,
        finish - start
    ))
    print()
    logfile = open(log_path, 'a+')
    logfile.write('Best Validation Accuracy {:.4f}'.format(sim_data["best_val_acc"]) + '\n\n\n')
    logfile.write('Test Accuracy {:.4f}'.format(sim_data["test_acc"]) + '\n\n\n')
    logfile.write(str(sim_data))
    logfile.close()
    return sim_data

