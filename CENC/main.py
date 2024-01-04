# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil

from loss import loss_coteaching
# from data_clothing1M import clothing_dataloader
import warnings
#warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,1'


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = 1)#0.5,0.75,1,1.25,1.5
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric, asymmetric]', default=None)
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100', default = 'mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument( '--data_aug', default=0, type=int, help="data augmentation (0 or 1)")

args = parser.parse_args()
'''
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=Model(input_size,output_size)
if torch.cuda.device_count()>1:
    print(f"Let's use{torch.cuda.device_count()}GPUs!")
    model=nn.DataParallel(model)
model.to(device)
'''
# Seed初始的权值参数初始化成随机数，减少一定程度上算法结果的随机性
torch.manual_seed(args.seed)#设置CPU随机数的种子
torch.cuda.manual_seed(args.seed)#设置GPU随机数的种子

# Hyper Parameters 超参数（调优参数）
batch_size = 128#一次训练选取的样本数
learning_rate = args.lr#学习率

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80#衰退的开始
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                        )
    
if args.dataset=='cifar10' and args.data_aug!=1:
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/cifar-10-batches-py',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                           )
    
    test_dataset = CIFAR10(root='./data/cifar-10-batches-py',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )

if args.dataset == 'cifar10' and args.data_aug == 1:
    input_channel = 3
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/cifar-10-batches-py',
                            download=True,
                            train=True,
                            transform=transforms.Compose(
                                [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914,
                                                       0.4822, 0.4465),
                                                      (0.2023, 0.1994,
                                                       0.2010))]),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    test_dataset = CIFAR10(root='./data/cifar-10-batches-py',
                           download=True,
                           train=False,
                           transform=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.4914,
                                                      0.4822, 0.4465),
                                                     (0.2023, 0.1994,
                                                      0.2010))]),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                           )

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200
    train_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )

    test_dataset = CIFAR100(root='./data/',
                                download=True,
                                train=False,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )

if args.forget_rate == 1:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate*args.noise_rate
# print(forget_rate)
noise_or_not = train_dataset.noise_or_not
# print(noise_or_not)
# Adjust learning rate and betas for Adam Optimizer调整Adam优化器的学习率和计算系数
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

# define drop rate schedule 更新forget rate
rate_schedule = np.ones(args.n_epoch)*forget_rate#返回给定形状和数据类型的新数组，其中元素的值设置为1
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)#创建等差数列start stop num ,c,轮

save_dir = args.result_dir +'/' +args.dataset+'/data_aug/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_'+args.noise_type+'_'+str(args.noise_rate)+'_forget_rate_'+str(args.forget_rate)+'_Tk_'+str(args.num_gradual)+'_c_'+str(args.exponent)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


#准确度计算 每次迭代中计算top1和topk，然后求平均
def accuracy(logit, target, topk=(1,)):#模型输出（batch_size×num_of_class），目标label（num_of_class向量），元组（分别向求top几）
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)#总元素的个数

    _, pred = output.topk(maxk, 1, True, True)# 返回最大的k个结果（按最大到小排序）
    pred = pred.t()# 转置
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2,model3,optimizer3):
    print ('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    pure_ratio_3_list = []

    train_total1=0
    train_correct1=0
    train_total2=0
    train_correct2=0
    train_total3=0
    train_correct3 = 0


    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)#最终的全连接层输出
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))#准确率
        train_total1+=1
        train_correct1+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2

        logits3 = model3(images)
        prec3, _ = accuracy(logits3, labels, topk=(1, 5))
        train_total3 += 1
        train_correct3 += prec3

        loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2,pure_ratio_3 = loss_coteaching(logits1, logits2,logits3, labels, rate_schedule[epoch], ind, noise_or_not)#计算损失函数
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)
        pure_ratio_3_list.append(100*pure_ratio_3)


        optimizer1.zero_grad()#清空过往梯度
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        loss_1.backward(retain_graph=True)#反向传播，计算当前梯度
        loss_2.backward(retain_graph=True)
        loss_3.backward()

        optimizer1.step()  # 根据梯度更新网络参数
        optimizer2.step()
        optimizer3.step()

        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Training Accuracy3: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f,Pure Ratio3 %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, prec3, loss_1.item(), loss_2.item(), loss_3.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list),np.sum(pure_ratio_3_list)/len(pure_ratio_3_list)))

    train_acc1=float(train_correct1)/float(train_total1)#正确率
    train_acc2=float(train_correct2)/float(train_total2)
    train_acc3 = float(train_correct3) / float(train_total3)
    return train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list, pure_ratio_3_list

# Evaluate the Model
def evaluate(test_loader, model1, model2, model3):
    print ('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    model3.eval()  # Change model to 'eval' mode
    correct3 = 0
    total3 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits3 = model3(images)
        outputs3 = F.softmax(logits3, dim=1)
        _, pred3 = torch.max(outputs3.data, 1)
        total3 += labels.size(0)
        correct3 += (pred3.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    acc3 = 100 * float(correct3) / float(total3)
    return acc1, acc2, acc3


def main():
    # Data Loader (Input Pipeline)
    print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print ('building model...')
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)# 输入通道数=3RGB图像 输出=分类数
    cnn1.cuda()
    print (cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)#优化器
    
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn2.cuda()
    print (cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    cnn3 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn3.cuda()
    print(cnn3.parameters)
    optimizer3 = torch.optim.Adam(cnn3.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0
    mean_pure_ratio3 = 0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 train_acc3 test_acc1 test_acc2 test_acc3 pure_ratio1 pure_ratio2 pure_ratio3\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    train_acc3=0
    # evaluate models with random weights
    test_acc1, test_acc2, test_acc3=evaluate(test_loader, cnn1, cnn2, cnn3)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %% Pure Ratio3 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' ' + str(train_acc3) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  +str(test_acc3) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) +' '  + str(mean_pure_ratio3) + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        cnn3.train()
        adjust_learning_rate(optimizer3, epoch)
        train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list, pure_ratio_3_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2,cnn3, optimizer3)
        # evaluate models
        test_acc1, test_acc2,test_acc3=evaluate(test_loader, cnn1, cnn2,cnn3)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        mean_pure_ratio3 = sum(pure_ratio_3_list) / len(pure_ratio_3_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%, Pure Ratio 3 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(train_acc3) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(test_acc3) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2)+' '+ str(mean_pure_ratio3) + "\n")

if __name__=='__main__':
    main()
