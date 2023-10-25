import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, correct_num, _ECELoss, RecallK, \
    DistillKL, mixup_data, adjust_lr, cutmix_data, CrossEntropyLoss_label_smooth
from data_loader.CIFAR_100 import get_cifar100_dataloaders
from data_loader.dataloader import load_dataset
from sync_batchnorm import convert_model

import time
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', default='CIFAR-100', type=str, help='Dataset directory')
parser.add_argument('--data_folder', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--arch', default='CIFAR_ResNet18', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[100, 150], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=200, type=int, help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='batch size')
parser.add_argument('--mixture-way', default='both', type=str, help='mixup|cutmix|both')
parser.add_argument('--kd-method', default='CIMGS-KD', type=str, help='mixup|cutmix|both')
parser.add_argument('--additional-aug', default='None', type=str, help='resume from checkpoint')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='temperature for KD distillation')
parser.add_argument('--alpha-cls', type=float, default=1, help='temperature for KD distillation')
parser.add_argument('--alpha-mix-cls', type=float, default=0., help='temperature for KD distillation')
parser.add_argument('--alpha-kd', type=float, default=1, help='temperature for KD distillation')
parser.add_argument('--T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='Dataset directory')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')

# global hyperparameter set
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# create files and folders
if not os.path.isdir('./result'):
    os.mkdir('./result')

if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

log_txt = './result/' + str(os.path.basename(__file__).split('.')[0]) \
          + '_dataset_' + args.dataset \
          + '_arch_' + args.arch \
          + '_KD_' + args.kd_method \
          + '_' + str(args.manual_seed) + '.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) \
          + '_dataset_' + args.dataset \
          + '_arch_' + args.arch \
          + '_KD_' + args.kd_method \
          + '_' + str(args.manual_seed)

print('dir for checkpoint:', log_dir)
with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')
        

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# ---------------------------------------------------------------------------------------

trainloader, valloader = load_dataset(name=args.dataset, root=args.data_folder, args=args)

print('Dataset: '+ args.dataset)
print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))
print('Number of classes: ' , trainloader.dataset.num_classes)
num_classes = trainloader.dataset.num_classes
C, H, W =  trainloader.dataset[0][0][0].size() if isinstance(trainloader.dataset[0][0], list) is True  else trainloader.dataset[0][0].size()

# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)

if args.kd_method == 'virtual_softmax':
    net = model(num_classes=num_classes, is_bias=False)
else:
    net = model(num_classes=num_classes)

print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
    % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, (1, C, H, W)) / 1e9))

del (net)


if args.kd_method == 'virtual_softmax':
    net = model(num_classes=num_classes, is_bias=False).cuda()
else:
    net = model(num_classes=num_classes).cuda()

net = torch.nn.DataParallel(net)
cudnn.benchmark = True



# Training
def train(epoch, criterion_list, optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_div = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    net.train()
    for batch_idx, (input, target) in enumerate(trainloader):
        batch_start_time = time.time()
        if isinstance(input, list) is False:
            input = input.cuda()
        target = target.cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        loss_div = torch.tensor([0.]).cuda()
        loss_cls = torch.tensor([0.]).cuda()

        batch_size = target.size(0)
        if args.kd_method == 'cross_entropy':
            if args.additional_aug == 'None' or args.additional_aug == 'cutout':
                logit = net(input)
                loss_cls += criterion_cls(logit, target)
            elif args.additional_aug == 'mixup':
                mixed_x, y_a, y_b, lam_mixup, _ = mixup_data(input.clone(), target, args=args, alpha=0.4)
                logit = net(mixed_x)
                loss_cls += criterion_cls(logit, y_a) * lam_mixup + criterion_cls(logit, y_b) * (1. - lam_mixup)
            elif args.additional_aug == 'cutmix':
                cutmix_x, target_a, target_b, lam_cutmix, _ = cutmix_data(input.clone(), target, args=args, alpha=1.)
                logit = net(cutmix_x)
                loss_cls += criterion_cls(logit, target_a) * lam_cutmix + criterion_cls(logit, target_b) * (1. - lam_cutmix)
            elif args.additional_aug == 'both':
                mixed_x, y_a, y_b, lam_mixup, _ = mixup_data(input.clone(), target, args=args, alpha=0.4)
                logit = net(mixed_x)
                loss_cls += criterion_cls(logit, y_a) * lam_mixup + criterion_cls(logit, y_b) * (1. - lam_mixup)
                cutmix_x, target_a, target_b, lam_cutmix, _ = cutmix_data(input.clone(), target, args=args, alpha=1.)
                logit = net(cutmix_x)
                loss_cls += criterion_cls(logit, target_a) * lam_cutmix + criterion_cls(logit, target_b) * (1. - lam_cutmix)
        
        elif args.kd_method == 'label_smooth':
            logit = net(input, target)
            loss_cls += CrossEntropyLoss_label_smooth(logit, target, num_classes=num_classes)
        elif args.kd_method == 'virtual_softmax':
            logit = net(input, target, loss_type='virtual_softmax')
            loss_cls += criterion_cls(logit, target)
        elif args.kd_method == 'Maximum_entropy':
            logit = net(input, target)
            entropy = (F.softmax(logit, dim=1) * F.log_softmax(logit, dim=1)).mean()
            loss_cls += criterion_cls(logit, target) + 0.5 * entropy

        elif args.kd_method == 'DKS':
            outputs = net(input)
            for j, output in enumerate(outputs):
                loss_cls += criterion_cls(output, target)
                for output_counterpart in outputs:
                    if output_counterpart is not output:
                        loss_div += criterion_div(output, output_counterpart.detach())
                    else:
                        pass
            logit = outputs[0]
        elif args.kd_method == 'BYOT':
            logits, features = net(input, feature=True)
            for i in range(len(logits)):
                loss_cls += criterion_cls(logits[i], target)
                if i != 0:
                    loss_div += criterion_div(logits[i], logits[0].detach())
            
            for i in range(1, len(features)):
                if i != 1:
                    loss_div += 0.5 * 0.1 * ((features[i] - features[1].detach()) ** 2).mean()
            
            logit = logits[0]   
        elif args.kd_method == 'DDGSD':
            input = torch.cat(input, dim=0)
            logit, features = net(input, feature=True)
            loss_cls += criterion_cls(logit, torch.cat([target, target], dim=0)) / 2
            loss_div += criterion_div(logit[:batch_size], logit[batch_size:].detach())
            loss_div += criterion_div(logit[batch_size:], logit[:batch_size].detach())
            loss_div += 5e-4 * (features[:batch_size].mean()-features[batch_size:].mean()) ** 2
            logit = (logit[batch_size:] + logit[:batch_size]) / 2
        elif args.kd_method == 'CS-KD':
            target = target[:batch_size//2]
            logit = net(input[:batch_size//2])
            with torch.no_grad():
                outputs_cls = net(input[batch_size//2:])
            loss_cls += criterion_cls(logit, target)
            loss_div += criterion_div(logit, outputs_cls.detach())
        elif args.kd_method == 'CIMGS-KD':
            lam = 0.5
            cutmix_input = cutmix_data(input.clone(), target, args=args, alpha=-1.)
            mix_input = (lam * input[:batch_size // 2, :] + (1 - lam) * input[batch_size // 2:, :]) / 2
            input = torch.cat([mix_input, cutmix_input, input], dim=0)
            logit = net(input)

            mixup_logits = logit[:batch_size//2, :]
            cutmix_logit = logit[batch_size//2: batch_size, :]
            class_ensemble_logits = (lam * logit[batch_size:batch_size+batch_size // 2] \
                                    + (1 - lam) * logit[batch_size+batch_size // 2: 2 *batch_size])
            logit = logit[batch_size:2 * batch_size]

            loss_cls += args.alpha_cls * criterion_cls(logit, target)

            
            loss_div += criterion_div(mixup_logits, class_ensemble_logits.detach())
            loss_div += criterion_div(class_ensemble_logits, mixup_logits.detach())
            loss_div += criterion_div(cutmix_logit, class_ensemble_logits.detach())
            loss_div += criterion_div(class_ensemble_logits, cutmix_logit.detach())

            loss_div = loss_div * args.alpha_kd
        else:
            raise ValueError('Unknown mode: {}'.format(args.distill))
        loss = loss_cls + loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)
        train_loss_div += loss_div.item() / len(trainloader)

        top1, top5 = correct_num(logit, target, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += target.size(0)
        
        
        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Acc:{:.4f}, Duration:{:.2f}'.format(epoch, batch_idx, len(trainloader), lr, top1_num.item() / total, time.time()-batch_start_time))
    
    print('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}'
                '\t train_loss_div:{:.5f}'
                '\n top1_acc: {:.4f} \t top5_acc:{:.4f}'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        train_loss_div, (top1_num/total).item(), (top5_num/total).item()))
    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\ntrain_loss:{:.5f}\t train_loss_cls:{:.5f}'
                '\t train_loss_div:{:.5f}'
                '\ntop1_acc: {:.4f} \t top5_acc:{:.4f} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls,
                        train_loss_div, (top1_num/total).item(), (top5_num/total).item()))


def test(epoch, criterion_list):
    test_loss = 0.
    test_loss_cls = 0.
    top1_num = 0
    top5_num = 0
    total = 0

    criterion_cls = criterion_list[0]

    if args.evaluate:
        criterion_ece = criterion_list[2]
        criterion_recall = criterion_list[3]
        logits_bank = []
        feature_bank = []
        label_bank = []

    net.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(valloader):
            input = input.float()
            input = input.cuda()
            target = target.cuda()

            if args.evaluate:
                logit, features = net(input, feature=True)
                if args.kd_method == 'DHM' or args.kd_method == 'BYOT':
                    logit = logit[0]
                    features = features[0]
                logits_bank.append(logit)
                feature_bank.append(features)
                label_bank.append(target)
            else:
                logit = net(input)
                if args.kd_method == 'DHM' or args.kd_method == 'BYOT':
                    logit = logit[0]
            loss_cls = criterion_cls(logit, target)
            loss = loss_cls

            test_loss += loss.item() / len(trainloader)
            test_loss_cls += loss_cls.item() / len(trainloader)

            top1, top5 = correct_num(logit, target, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += target.size(0)

    
    with open(log_txt, 'a+') as f:
        f.write('test_loss:{:.5f}\t test_loss_cls:{:.5f}'
                '\t top1_acc:{:.4f} \t top5_acc:{:.4f} \n'
                .format(test_loss, test_loss_cls, (top1_num/total).item(), (top5_num/total).item()))
        print('test epoch:{}\t top1_acc:{:.4f} \t top5_acc:{:.4f}'.format(epoch, (top1_num/total).item(), (top5_num/total).item()))

    if args.evaluate:
        logits_bank = torch.cat(logits_bank, dim=0)
        feature_bank = torch.cat(feature_bank, dim=0)
        label_bank = torch.cat(label_bank, dim=0)
        ece = criterion_ece(logits_bank, label_bank)
        recallk = criterion_recall(feature_bank, label_bank)
        top1_acc = (top1_num/total).item()
        top5_acc = (top5_num/total).item()
        with open(log_txt, 'a+') as f:
            f.write('test_metrics: top1_acc:{:.4f} top5_acc:{:.4f} ece:{:.4f} recallk:{:.4f} \n'.format(top1_acc, top5_acc, ece, recallk))
        print('test_metrics: top1_acc:{:.4f} top5_acc:{:.4f} ece:{:.4f} recallk:{:.4f}'.format(top1_acc, top5_acc, ece, recallk))
        return top1_acc

    else:
        return (top1_num/total).item()



if __name__ == '__main__':
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.T)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)
    criterion_list.append(_ECELoss(n_bins=20))
    criterion_list.append(RecallK(K=1))
    criterion_list.cuda()

    if args.evaluate:
        print('load trained weights from '+ args.checkpoint_dir + '/' + model.__name__ + '_best.pth.tar')
        checkpoint = torch.load(args.checkpoint_dir + '/' +  model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        top1_acc = test(start_epoch, criterion_list)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            print('Resume from '+os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']+1

        start_epoch = 0
        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_list)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, model.__name__ + '_best.pth.tar'))

        print('Evaluate the best model:')
        args.evaluate = True
        checkpoint = torch.load(args.checkpoint_dir + '/' +  model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_list)

        with open(log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + log_txt + ' ' + args.checkpoint_dir)

