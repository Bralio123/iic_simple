import torch
import torch.nn as nn
import torch.optim as optim


import torchvision
import torchvision.datasets as dataset

import matplotlib.pyplot as plt
import numpy as np

from skimage.util import montage

import random
import sys
import os
import time

root = './data'
N = 10
batch_size = 350
lr=0.0001
num_epochs = 3200
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
EPS = sys.float_info.epsilon
PATH = r'D:\models\iic\saved'

def disp_first_batch(t, tf):
    
    train_batch = next(iter(t))
    train_tf_batch = next(iter(tf))
    
    join_batch = np.concatenate([train_batch[0].detach().cpu().numpy(), train_tf_batch[0].detach().cpu().numpy()], axis=0)
    
    plt.imshow(montage(join_batch[:, 0, :, :]))



def transform_list():
    
    tf1_list = []
    tf3_list = []
    tf2_list = []
     
    crop_size = 20
    input_size = 28
     
    tf1_crop_fn = torchvision.transforms.RandomChoice([
       torchvision.transforms.RandomCrop(crop_size),
       torchvision.transforms.CenterCrop(crop_size)
     ])
    
    tf1_list += [tf1_crop_fn]
    
    tf1_list += [torchvision.transforms.Resize(input_size),
               torchvision.transforms.ToTensor()]
    
    
    tf3_list += [torchvision.transforms.CenterCrop(crop_size)]
    tf3_list += [torchvision.transforms.Resize(input_size),
               torchvision.transforms.ToTensor()]
    
    tf2_list += [torchvision.transforms.RandomApply(
        [torchvision.transforms.RandomRotation(25)], p=0.5)]
    
    tf2_crop_fn = torchvision.transforms.RandomChoice([
       torchvision.transforms.RandomCrop(crop_size),
       torchvision.transforms.RandomCrop(crop_size + 4),
       torchvision.transforms.RandomCrop(crop_size - 4)
     ])
    
    tf2_list += [tf2_crop_fn]
    
    tf2_list += [torchvision.transforms.Resize(input_size)]
    
    tf2_list += [
      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.4, hue=0.125), 
      torchvision.transforms.ToTensor()]
      
    tf1_list +=[torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    tf2_list +=[torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    tf3_list +=[torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    
    tf1 = torchvision.transforms.Compose(tf1_list)
    tf2 = torchvision.transforms.Compose(tf2_list)
    tf3 = torchvision.transforms.Compose(tf3_list)
    
    return tf1, tf2, tf3
    

class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), #in; out; kernel; stride; padding
            nn.ReLU(),
            nn.BatchNorm2d(64), #number of kernels            
            
            nn.Conv2d(64, 128, 3, 1, 1), #in; out; kernel; stride; padding
            nn.ReLU(),
            nn.BatchNorm2d(128), #number of kernels            
            nn.MaxPool2d(2), #kernel; stride
            
            nn.Conv2d(128, 256, 3, 1, 1), #in; out; kernel; stride; padding
            nn.ReLU(),
            nn.BatchNorm2d(256), #number of kernels            
            
            nn.Conv2d(256, 512, 3, 1, 1), #in; out; kernel; stride; padding
            nn.ReLU(),
            nn.BatchNorm2d(512), #number of kernels       
            nn.MaxPool2d(2)
            
        )
        
        self.final = nn.Sequential(
            nn.Linear(512 * 7 * 7, N),
            nn.Softmax(dim=1)
        )
        
    def init_weigths(self):   
        
        for m in self.modules:
            print(m)
        
        
        return -1
        
    def forward(self, x):
        bs = x.size(0)
        
        x = self.features(x)           
        x = x.view(bs, -1)        
        x = self.final(x)

        return x
    
def custom_loss(outs, outs_tf):
      
    # LOSS CALCULATED
    p = torch.zeros((N, N), dtype=torch.float, device=device)
    for sample_idx in range(batch_size):
        temp_out = outs[sample_idx].view(10, 1)
        temp_out_tf = outs_tf[sample_idx].view(1, 10)                    
        p = p + (temp_out * temp_out_tf)
    
    p = p / batch_size
    
    p_i = p.sum(dim=0)
    p_j = p.sum(dim=1)
    
    conditional_entropy = 0
    for i in range(N):
        for j in range(N):
            conditional_entropy += p[i][j] * (- torch.log(p[i][j]) - torch.log(p_i[i]))                        
    conditional_entropy = - conditional_entropy
    
    entropy_z = 0        
    for i in range(N):                    
        entropy_z += p_i[i] * torch.log(p_i[i])
    entropy_z = - entropy_z                    
        
    entropy_z_tf = 0        
    for j in range(N):                    
        entropy_z_tf += p_j[j] * torch.log(p_j[j])                    
        
    entropy_z_tf = - entropy_z_tf
    
    loss_hard = 0
    for i in range(N):
        for j in range(N):
            loss_hard+= p[i][j] * (- torch.log(p[i][j]) - torch.log(p_i[i]) - torch.log(p_j[j]))
    
    loss_derived = entropy_z - conditional_entropy
    
    return loss_derived, loss_hard
            
    
def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))
    
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric
    
    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS
    
    loss = - p_i_j * (torch.log(p_i_j) \
                - lamb * torch.log(p_j) \
                - lamb * torch.log(p_i))
    
    loss = loss.sum()
    
    loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))
    
    loss_no_lamb = loss_no_lamb.sum()
    
    return loss, loss_no_lamb


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def pairwise_shuffle(train, aug):
    print('shuffling datasets...')
    shuff_list = list(zip(train, aug))
    random.shuffle(shuff_list)
    train_set, aug_set = zip(*shuff_list)
    
    return train_set, aug_set


def train(m, train_loader, aug_loader):
    m = conv_net()
    m.to(device)
    optimiser = optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.99))

    epoch_losses = []   
    m.train()
    for epoch in range(num_epochs):         
        step = 0
        t_epoch = time.time()
        
        batch_losses = []    
        for idx, data in enumerate(zip(train_loader, aug_loader)):
            
            x_loader = data[0]
            x_tf_loader = data[1]
            
            xs = x_loader[0]            
            xs_tf = x_tf_loader[0]
            xs = xs.to(device)
            xs_tf = xs_tf.to(device)

            #zero the parameter gradients
            optimiser.zero_grad()
            
            outs = m(xs)
            outs_tf = m(xs_tf)
            
            #LOSS GIT REPO
            _, loss = IID_loss(outs, outs_tf)
            
            #LOSS PAPER                
            P = (outs.unsqueeze(2) * outs_tf.unsqueeze(1)).sum(dim=0)
            P = ((P + P.t()) / 2) / P.sum()
            P[(P < EPS).data] = EPS
            Pi = P.sum(dim=1).view(N, 1).expand(N, N)
            Pj = P.sum(dim=0).view(1, N).expand(N, N)
            loss_paper = (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()
            
            # BACKWARD PROPAGATION
            loss.backward()            
            optimiser.step()
            
            batch_losses.append(loss.item())
            
            #print every 10 mini_batches
            if idx % 10 == 0:
                print('avg loss after {} batches: {}'.format(idx, sum(batch_losses) / len(batch_losses)))
        
        curr_epoch_loss = sum(batch_losses) / len(batch_losses)
        
        if not epoch_losses:
            min_loss = 0
        else:
            min_loss = min(epoch_losses)            
        
        epoch_losses.append(curr_epoch_loss)
        
        model_to_save = 'model_' + str(epoch) + '.pth'
        
        #save model at current epoch
        if curr_epoch_loss <= min_loss:                    
            print('saving model...')
            torch.save(m.state_dict(), os.path.join(PATH, 'epoch-{}.pt'.format(epoch)))            
            
        print('avg loss {} at epoch {} took {} '.format(curr_epoch_loss, epoch, time.time() - t_epoch))
        
    return epoch_losses    

def evaluate(model, test_loader, print_conf_matrix=True):
    model.eval()
    
    conf_matrix = torch.zeros(10, 10).int()
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
    if print_conf_matrix:
        print(conf_matrix)
        
        
    model.train()
    return acc    

if __name__ == '__main__':
    
    tf1, tf2, tf3 = transform_list()
    
    train_set = dataset.MNIST(root=root, transform=tf1, train=True, download=True)
    aug_set = dataset.MNIST(root=root, transform=tf2, train=True, download=True)
    test_set = dataset.MNIST(root=root, transform=tf3, train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=False,            
        )
    
    aug_loader = torch.utils.data.DataLoader(
            dataset=aug_set,
            batch_size=batch_size,
            shuffle=False,            
        )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=20,                
            shuffle=False
         )
    
    model = conv_net()
    model.load_state_dict(torch.load(r'D:\models\iic\saved\epoch-3150.pt'))
    model = model.to(device)
    
    acc = evaluate(model, test_loader)
    