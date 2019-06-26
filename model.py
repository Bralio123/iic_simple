from __future__ import print_function
from skimage.util import montage
from batchup import data_source
from batchup.datasets import mnist
from PIL import Image
from torch.autograd import Variable

import time
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

vgg = models.vgg16_bn()
resnet = models.resnet101()
num_epochs = 500
lr = 0.001
batch_size = 128
num_channels = 1
im_H = 28
im_W = 28
C=10
lamb=1.0
EPS=0.001

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

def disp_first_batch(t, tf):
    train_batch = next(iter(t))
    train_tf_batch = next(iter(tf))
    
    join_batch = np.concatenate([train_batch[0].detach().cpu().numpy(), train_tf_batch[0].detach().cpu().numpy()], axis=0)
    
    plt.imshow(montage(join_batch[:, 0, :, :]))

class resNetFull(nn.Module):
    def __init__(self, num_classes):
        super(resNetFull, self).__init__()
        self.in_chann = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.features = resnet
        self.final = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=1000, out_features=512, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=512, out_features=128, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=128, out_features=num_classes, bias=True),
                                   nn.Softmax(dim=1)
                                 )
        
        
    def forward(self, x):
        x = self.in_chann(x)
        x = self.features(x)
        x = self.final(x)
        print(x)
        return x        
    
class vggFull(nn.Module):
    def __init__(self, num_classes):
        super(vggFull, self).__init__()
        self.in_chann = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.features = nn.Sequential(*list(vgg.children())[:-1])
        self.final = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=4096, out_features=1000, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=1000, out_features=num_classes, bias=True),
                                   nn.Softmax(dim=1)
                                 )
        
    def forward(self, x):  
        x = self.in_chann(x)              
        x = self.features(x)
        x = x.view(-1, 25088) #512 * 7 * 7
        x = self.final(x)
        return x    

def IID_loss(x_out, x_tf_out, lamb=1.0):
  
  # has had softmax applied
  _, k = x_out.size()
  p_i_j = compute_joint(x_out, x_tf_out)
  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
  
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

if __name__ == "__main__":    

    net = resNetFull(10)
    net.cuda()
    net.train()
    optimiser = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))

    mag = np.array([[[0.17, 0.17, 2.0/28.0],
                     [0.17, 0.17, 2.0/28.0]]])
    
    identity = np.array([[[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]]])
    
    
    random_affine = np.repeat(identity, batch_size, axis=0)
    
    for idx in range(batch_size):        
        random_affine[idx] = random_affine[idx] + np.random.normal(size=(2,3)) * mag
        
    t_affine = torch.tensor(random_affine, dtype=torch.float, device=device)
    
    ds = mnist.MNIST(n_val=10000)
    
    ds2 = data_source.ArrayDataSource([ds.train_X, ds.train_y])
    
    train_losses = []        
    avg_train_losses = []    
    avg_valid_losses = []
    initial_loss = float('inf')
    
    for epoch in range(num_epochs): 
        step = 0
        t_epoch = time.time()
        for batch_x, _ in ds2.batch_iterator(batch_size=batch_size, shuffle=True):
            curr_batch_sz, _, _, _ = batch_x.shape
            if curr_batch_sz == batch_size:
                batch_x = Variable(torch.from_numpy(batch_x)).cuda()
                grid = F.affine_grid(t_affine, batch_x.shape)
                x_aug = F.grid_sample(batch_x, grid)
                x_aug = Variable(x_aug).cuda()
                            
                #zero the parameter gradients
                optimiser.zero_grad()
                
                outs = net(batch_x)
                outs_tf = net(x_aug) 
                
                loss, loss_no_lamb = IID_loss(outs, outs_tf)
                
                print('loss_paper: {}'.format(loss))
                
                """
                p = torch.zeros((C, C), dtype=torch.float, device=device)
                for sample_idx in range(batch_size):
                    temp_out = outs[sample_idx].view(10, 1)
                    temp_out_tf = outs_tf[sample_idx].view(1, 10)                    
                    p = p + (temp_out * temp_out_tf)
                
                p = p / batch_size
                
                p_i = p.sum(dim=0)
                p_j = p.sum(dim=1)
                
                conditional_entropy = 0
                for i in range(C):
                    for j in range(C):
                        conditional_entropy += p[i][j] * (- torch.log(p[i][j]) - torch.log(p_i[i]))                        
                conditional_entropy = - conditional_entropy                        
                
                entropy_z = 0        
                for i in range(C):                    
                    entropy_z += p_i[i] * torch.log(p_i[i])
                entropy_z = - entropy_z                    
                    
                entropy_z_tf = 0        
                for j in range(C):                    
                    entropy_z_tf += p_j[j] * torch.log(p_j[j])                    
                    
                entropy_z_tf = - entropy_z_tf
                
                loss = 0
                for i in range(C):
                    for j in range(C):
                        loss+= p[i][j] * (- torch.log(p[i][j]) - torch.log(p_i[i]) - torch.log(p_j[j]))
                
                loss_derived = entropy_z - conditional_entropy
                
                loss_derived = -loss_derived
                print('entropy_z_tf: {}'.format(entropy_z_tf))
                print('entropy_z: {}'.format(entropy_z))
                print('conditional_entropy: {}'.format(conditional_entropy))
                print('loss: {}'.format(loss_derived))
                
                """    
                
                
                loss.backward()            
                optimiser.step()
                train_losses.append(loss.item())
                
                if np.isnan(float(loss)):
                    print('NaN encountered: epoch {} batch {}'.format(epoch, step))
                
                if step % 100 == 99: # print every 100 batches            
                    print('Loss at batch {} ; epoch {}: {}'.format(step, epoch, loss.item()))
                    
                step = step + 1                    
          
        avg_loss = sum(train_losses) / len(train_losses)
        avg_train_losses.append(avg_loss)
        
        if not avg_train_losses:
            min_loss = initial_loss
        else:                    
            min_loss = np.min(avg_train_losses) 
            
        print("Finish epoch {}, time elapsed {}, Train Loss: {}".format(epoch, time.time() - t_epoch, avg_loss)) 

        #if the new avg loss is better than save this model
        if(min_loss >= avg_loss):
            print('Better loss; saving model at epoch: {:d}'.format(epoch))
            print('-------------------------------------------------------')
            
            model_dir= r'd:\models\iic\ADAM_LR10e5'             
            model_filename = 'model_segnet_norm_{}.pth'.format(epoch)
            
            path = os.path.join(model_dir, model_filename)

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': avg_train_losses[epoch],
                    }, path)   

        #save every 50 epochs arbitrarly
        if(epoch % 50 == 49):
            print('Saving model at epoch: {:d}'.format(epoch))
            
            #model_dir= r'd:\models\segnet_models\pre_trained'        
            model_dir= r'c:\temp'        
            model_filename = 'model_segnet_norm_{}.pth'.format(epoch)
            
            path = os.path.join(model_dir, model_filename)

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': avg_train_losses[epoch],
                    }, path)   
           