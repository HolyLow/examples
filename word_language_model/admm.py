from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import sys
import numpy as np
from numpy import linalg as LA
import yaml
import datetime

from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Logger(object):
    
    def __init__(self):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter('../log_{}'.format(datetime.datetime.now()))
        
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        
        self.writer.add_scalar(tag,value,step)
    '''
    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
    '''
    '''
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
    '''
logger = Logger()
class Config:
    def __init__(self,args,model):
        self._prune_ratios = {} # kv pairs {good_name:prune_ratio}
        self.prune_ratios = {} # kv pairs  {code_name:prune_ratio}
        self.rhos = {}
        self.sparsity_type = None
        self.name_encoder = {} # meaningful name -> code name
        self.model = model
        self.init(args,model)
        self.masks = None
        self.zero_masks = None
        for k in self.prune_ratios.keys():
            self.rhos[k] = self.rho      # this version we assume all rhos are equal
        print ("debug")
        print (self.prune_ratios)
    def _read_file(self,args,model):
        """
        read config file
        """
        try:
            with open(args.config_file,"r") as stream:            
                raw_dict = yaml.load(stream)

                self._prune_ratios = raw_dict[args.arch]['prune_ratios'] # read prune ratio from yaml
                self.rho = args.rho                
                self.sparsity_type = args.sparsity_type
        except yaml.YAMLError as exc:
            print(exc)
    def _extract_layer_names(self):
         """
         Store layer name of different types in arrays for indexing
         """
         self.conv_names = []
         self.fc_names = []
         self.bn_names = []
         names = []
         for name, W in self.model.named_modules():
             names.append(name)
         print (names)
         for name,W in self.model.named_modules():             
             name+='.weight'  # name in named_modules looks like module.features.0. We add .weight into it
             if isinstance(W,nn.Conv2d):
                 self.conv_names.append(name)
             if isinstance(W,nn.BatchNorm2d):
                 self.bn_names.append(name)
             if isinstance(W,nn.Linear):
                 self.fc_names.append(name)
                
    def _encode(self,name):
         """
         Examples:
         conv1.weight -> conv           1                weight
                         conv1-> prefix   weight->postfix        
                         conv->layer_type  1-> layer_id + 1  weight-> postfix
         Use buffer for efficient look up  
         """
         prefix,postfix = name.split('.')
         dot_position = prefix.find('.')
         layer_id = ''
         for s in prefix:
             if s.isdigit():
                 layer_id+=s
         id_length = len(layer_id)         
         layer_type = prefix[:-id_length]
         layer_id = int(layer_id)-1
         if layer_type =='conv':
             self.name_encoder[name] = self.conv_names[layer_id]
         elif layer_type =='fc':
             self.name_encoder[name] =  self.fc_names[layer_id]
         elif layer_type =='bn':
             self.name_encoder[name] =  self.bn_names[layer_id]             

    def init(self,args,model):
         self._read_file(args,model)
         self._extract_layer_names()
         for good_name,ratio in self._prune_ratios.items():             
             self._encode(good_name)
         for good_name,ratio in self._prune_ratios.items():
             self.prune_ratios[self.name_encoder[good_name]] = ratio

class ADMM:
     def __init__(self,model,file_name):
          self.ADMM_U = {}
          self.ADMM_Z = {}
          self.ADMM_alpha = {} # used for quantization only
          self.ADMM_Q = {}# used for quantization only
          self.model = model
          self.prune_ratios = None    #code name -> prune ratio
          self.init(file_name,model)

     def init(self,config,model):
          """
          Args:
              config: configuration file that has settings for prune ratios, rhos
          called by ADMM constructor. config should be a .yaml file          

          """          
          self.prune_ratios = config.prune_ratios
          self.rhos = config.rhos
          
          self.sparsity_type = config.sparsity_type
          for (name,W) in model.named_parameters():
              if name not in config.prune_ratios:
                  continue
              self.ADMM_U[name] = torch.zeros(W.shape).cuda() # add U 
              self.ADMM_Z[name] = torch.Tensor(W.shape).cuda() # add Z
                        


               
def random_pruning(args,weight,prune_ratio):
     weight = weight.cpu().detach().numpy()            # convert cpu tensor to numpy
     
     
     if (args.sparsity_type == "filter"):
          shape = weight.shape
          weight2d = weight.reshape(shape[0],-1)
          shape2d = weight2d.shape
          indices = np.random.choice(shape2d[0],int(shape2d[0]*prune_ratio),replace = False)                    
          weight2d[indices,:] = 0
          weight = weight2d.reshape(shape)
          expand_above_threshold = np.zeros(shape2d,dtype=np.float32)
          for i in range(shape2d[0]):
               expand_above_threshold[i,:] = i not in indices
          weight = weight2d.reshape(shape)
          expand_above_threshold = expand_above_threshold.reshape(shape)     
          return torch.from_numpy(expand_above_threshold).cuda(),torch.from_numpy(weight).cuda()
     else:
          raise Exception("not implemented yet")


def L1_pruning(args,weight,prune_ratio):
     """
     projected gradient descent for comparison

     """
     percent = prune_ratio * 100               
     weight = weight.cpu().detach().numpy()            # convert cpu tensor to numpy     
     shape = weight.shape
     weight2d = weight.reshape(shape[0],-1)
     shape2d = weight2d.shape
     row_l1_norm = LA.norm(weight2d,1,axis = 1)
     percentile = np.percentile(row_l1_norm,percent)
     under_threshold = row_l1_norm <percentile
     above_threshold = row_l1_norm >percentile
     weight2d[under_threshold,:] = 0          
     above_threshold = above_threshold.astype(np.float32)
     expand_above_threshold = np.zeros(shape2d,dtype=np.float32)
     for i in range(shape2d[0]):
          expand_above_threshold[i,:] = above_threshold[i]
     weight = weight.reshape(shape)
     expand_above_threshold = expand_above_threshold.reshape(shape)
     return torch.from_numpy(expand_above_threshold).cuda(),torch.from_numpy(weight).cuda()     

def weight_quantization(args,ADMM,model):
    """
    Assuming weights are close to centrodis enough, do projection
    
    """
    for name,W in model.named_parameters():
        if name in ADMM.number_bits:
            weight = W.cpu().detach().numpy().flatten()
            shape = weight.shape
            num_centroids = 2**(config.number_bits[name]-1)
            centroid_candidates = []
            for j in range(num_centroids):
                centroid_candidates.append(2**j)
                centroid_candidates.append(-2**j)            
            for i in range(weight.size):
                minDist = 10000
                index = -1
                for j in range(num_centroids):
                    temp = (weight[i])/ADMM.ADMM_alpha[name]
                    dist = abs(temp - centroid_candidates[j])
                    if (dist < minDist):
                        minDist = dist
                        index = j
                weight[i] = centroid_candidates[index]*ADMM.ADMM_alpha[name]
            weight = weight.reshape(shape)
            W.data = torch.from_numpy(weight).cuda()
def Q_alpha_update(args,weight,Q,U,alpha,bits):
    """
    weight quantization 
    Args:
        weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height      
        bits (int between 1 - 8): target bit representation of weights
    Returns:
        
    """
    
    if args.sparsity_type != "quantization":
        raise Exception("Wrong sparsity type")

    if bits == None or bits<1 :
        raise Exception ("bits have to be 1-8 [int]")
    shape = weight.shape
    # flatten weight, Q, U so we can use a single for loop
    weight = weight.cpu().detach().numpy().flatten()
    Q = Q.cpu().detach().numpy().flatten() 
    U = U.cpu().detach().numpy().flatten()
    if alpha == None:
        pass
    else:
        pass

    num_centroids = 2**(bits-1)
    
    centroid_candidates = []
    for i in range(num_centroids):
        centroid_candidates.append(2**i)
        centroid_candidates.append(-2**i)
    # centroid_candidates.append(0) we don't always include 0
    num_iterations = 20 # 20 is sufficient for Q,alpha to converge
    for p in range(num_iterations):
        for i in range(weight.size):
            minDist = 10000
            index = -1
            for j in range(len(centroid_candidates)):
                temp = (weight[i]+U[i])/alpha
                dist = abs(temp-centroid_candidates[j])
                if (dist < minDist):
                    minDist = dist
                    index = j
            Q[i] = centroid_candidates[index]
        alpha = 0.0
        QtQ = 0.0
        for i in range(weight.size):
            alpha+= (weight[i]+U[i])*Q[i]
            QtQ += Q[i]*Q[i]
        alpha = alpha/QtQ

    Q = Q.reshape(shape)
    return torch.from_numpy(Q).cuda(),torch.from_numpy(alpha).cuda()
def weight_pruning(args,weight,prune_ratio):
     """ 
     weight pruning [irregular,column,filter]
     Args: 
          weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
          prune_ratio (float between 0-1): target sparsity of weights
     
     Returns:
          mask for nonzero weights used for retraining
          a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero 

     """     

     weight = weight.cpu().detach().numpy()            # convert cpu tensor to numpy     
    
     percent = prune_ratio * 100          
     if (args.sparsity_type == "irregular"):
         weight_temp = np.abs(weight)   # a buffer that holds weights with absolute values     
         percentile = np.percentile(weight_temp,percent)   # get a value for this percentitle
         under_threshold = weight_temp<percentile     
         above_threshold = weight_temp>percentile     
         above_threshold = above_threshold.astype(np.float32) # has to convert bool to float32 for numpy-tensor conversion     
         weight[under_threshold] = 0     
         return torch.from_numpy(above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (args.sparsity_type == "column"):
          shape = weight.shape          
          weight2d = weight.reshape(shape[0],-1)
          shape2d = weight2d.shape
          column_l2_norm = LA.norm(weight2d,2,axis = 0)
          percentile = np.percentile(column_l2_norm,percent)
          under_threshold = column_l2_norm<percentile
          above_threshold = column_l2_norm>percentile
          weight2d[:,under_threshold] = 0
          above_threshold = above_threshold.astype(np.float32)
          expand_above_threshold = np.zeros(shape2d,dtype=np.float32)          
          for i in range(shape2d[1]):
               expand_above_threshold[:,i] = above_threshold[i]
          expand_above_threshold = expand_above_threshold.reshape(shape)
          weight = weight2d.reshape(shape)          
          return torch.from_numpy(expand_above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (args.sparsity_type =="filter"):
          shape = weight.shape
          weight2d = weight.reshape(shape[0],-1)
          shape2d = weight2d.shape
          row_l2_norm = LA.norm(weight2d,2,axis = 1)
          percentile = np.percentile(row_l2_norm,percent)
          under_threshold = row_l2_norm <percentile
          above_threshold = row_l2_norm >percentile
          weight2d[under_threshold,:] = 0          
          above_threshold = above_threshold.astype(np.float32)
          expand_above_threshold = np.zeros(shape2d,dtype=np.float32)          
          for i in range(shape2d[0]):
               expand_above_threshold[i,:] = above_threshold[i]

          weight = weight2d.reshape(shape)
          expand_above_threshold = expand_above_threshold.reshape(shape)
          return torch.from_numpy(expand_above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (args.sparsity_type =="bn_filter"):
          ## bn pruning is very similar to bias pruning
          weight_temp = np.abs(weight)
          percentile = np.percentile(weight_temp,percent)
          under_threshold = weight_temp<percentile     
          above_threshold = weight_temp>percentile     
          above_threshold = above_threshold.astype(np.float32) # has to convert bool to float32 for numpy-tensor conversion     
          weight[under_threshold] = 0     
          return torch.from_numpy(above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (args.sparsity_type == "balanced_row"):
          shape = weight.shape()
          dim = 1
          weight = torch.from_numpy(weight).reshape(shape[0], -1)
          rank = int(prune_ratio * weight.shape[dim])
          weight_abs = weight.abs().clone().cpu()
          threshold, position = weight_abs.kthvalue(rank, dim, True)
          threshold = threshold.expand(weight.shape)
          mask = weight_abs.lt(threshold).cuda()   # values smaller than threshold will be tagged
          weight.masked_fill_(mask, 0.0)      # mask tagged data to zero
          weight.reshape(shape)
          return mask.astype(np.float32).reshape(shape).cuda(), weight.cuda()
     else:
          raise SyntaxError("Unknown sparsity type")
                                         
def test_sparsity(args,config,model):
     """
     test sparsity for every involved layer and the overall compression rate

     """
     total_zeros = 0
     total_nonzeros = 0

     if args.masked_progressive and (args.sparsity_type == 'filter' or args.sparsity_type =='column'):
         ### test both column and row sparsity
        print ("***********checking column sparsity*************")
        for name,W in model.named_parameters():               
            if  name not in config.conv_names:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0],-1)
            column_l2_norm = LA.norm(W2d,2,axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm !=0)

            print ("column sparsity of layer {} is {}".format(name,zero_column/(zero_column+nonzero_column)))
        print ("***********checking filter sparsity*************")            
        for name,W in model.named_parameters():
             if name not in config.conv_names:
                 continue
             W = W.cpu().detach().numpy()
             shape = W.shape
             W2d = W.reshape(shape[0],-1)
             row_l2_norm = LA.norm(W2d,2,axis=1)
             zero_row = np.sum(row_l2_norm == 0)
             nonzero_row = np.sum(row_l2_norm !=0)
             print ("filter sparsity of layer {} is {}".format(name,zero_row/(zero_row+nonzero_row)))
        print ("************checking overall sparsity in conv layers*************")
        for name,W in model.named_parameters():               
            if  name not in config.conv_names:
                continue
            W = W.cpu().detach().numpy()            
            total_zeros +=np.sum(W==0)
            total_nonzeros +=np.sum(W!=0)
        print ('only consider conv layers, compression rate is {}'.format((total_zeros+total_nonzeros)/total_nonzeros))
        return
    
     if args.sparsity_type == "irregular":
         for name,W in model.named_parameters():
              if 'bias' in name:
                   continue
              W = W.cpu().detach().numpy()
              zeros = np.sum(W==0)
              total_zeros+=zeros
              nonzeros = np.sum(W!=0)
              total_nonzeros+=nonzeros
              print ("sparsity at layer {} is {}".format(name,zeros/(zeros+nonzeros)))
         total_weight_number = total_zeros+total_nonzeros
         print ('overal compression rate is {}'.format(total_weight_number/total_nonzeros))
     elif args.sparsity_type == "column":
        for name,W in model.named_parameters():               
            if  name not in config.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0],-1)
            column_l2_norm = LA.norm(W2d,2,axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm !=0)
            total_zeros +=np.sum(W==0)
            total_nonzeros +=np.sum(W!=0)
            print ("column sparsity of layer {} is {}".format(name,zero_column/(zero_column+nonzero_column)))
        print ('only consider conv layers, compression rate is {}'.format((total_zeros+total_nonzeros)/total_nonzeros))          
     elif args.sparsity_type == "filter":
         for name,W in model.named_parameters():
             if name not in config.prune_ratios:
                 continue
             W = W.cpu().detach().numpy()
             shape = W.shape
             W2d = W.reshape(shape[0],-1)
             row_l2_norm = LA.norm(W2d,2,axis=1)
             zero_row = np.sum(row_l2_norm == 0)
             nonzero_row = np.sum(row_l2_norm !=0)
             total_zeros +=np.sum(W==0)
             total_nonzeros +=np.sum(W!=0)
             print ("filter sparsity of layer {} is {}".format(name,zero_row/(zero_row+nonzero_row)))
         print ('only consider conv layers, compression rate is {}'.format((total_zeros+total_nonzeros)/total_nonzeros))
     elif args.sparsity_type == "bn_filter":
          for i,(name,W) in enumerate(model.named_parameters()):
               if name not in config.prune_ratios:
                    continue
               W = W.cpu().detach().numpy()
               zeros = np.sum(W==0)
               nonzeros = np.sum(W!=0)
               print ("sparsity at layer {} is {}".format(name,zeros/(zeros+nonzeros)))


def admm_initialization(args,ADMM,model):
     if not args.admm:
          return
     for name,W in model.named_parameters():
          if name in ADMM.prune_ratios:
               _,updated_Z = weight_pruning(args,W,ADMM.prune_ratios[name]) # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
               ADMM.ADMM_Z[name] = updated_Z
                   

def admm_update(args,ADMM,model,device,train_loader,optimizer,epoch,data,batch_idx):
     if not args.admm:
         return
     # sometimes the start epoch is not zero. It won't be valid if the start epoch is not 0
     if epoch != 0 and epoch % args.admm_epoch == 0 and batch_idx == 0:            
         for name,W in model.named_parameters():
             if args.sparsity_type !="quantization":
                 if name not in ADMM.prune_ratios:
                     continue
                   
                 if args.verbose and args.sparsity_type!="quantization":
                     Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()

                 ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name] # Z(k+1) = W(k+1)+U[k]

                 _, _Z = weight_pruning(args,ADMM.ADMM_Z[name],ADMM.prune_ratios[name]) #  equivalent to Euclidean Projection
                 ADMM.ADMM_Z[name] = _Z

                 ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name]+ ADMM.ADMM_U[name] # U(k+1) = W(k+1) - Z(k+1) +U(k)
                 
                 if (args.verbose):
                    log_loss('layer:{} W(k+1)-Z(k+1)'.format(name),torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)),epoch)
                    log_loss('layer:{} Z(k+1)-Z(k)'.format(name),torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)),epoch)
             else:
                 if name not in ADMM.number_bits:
                     continue
                 _Q,_alpha = Q_alpha_update(args,W,self.ADMM_Q,self.ADMM_U,self.ADMM_alpha,ADMM.number_bits[name])
                 self.ADMM_Q = _Q
                 self.ADMM_alpha = _alpha
def append_admm_loss(args,ADMM,model,ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    ''' 
    admm_loss = {}
    
    if args.admm:
        if args.sparsity_type !="quantization":
            for name,W in model.named_parameters():  ## initialize Z (for both weights and bias)
                if name not in ADMM.prune_ratios:
                    continue

                admm_loss[name] = 0.5*ADMM.rhos[name]*(torch.norm(W-ADMM.ADMM_Z[name]+ADMM.ADMM_U[name],p=2)**2)
        else:
            for name,W in model.named_parameters():
                if name not in ADMM.number_bits:
                    continue
                admm_loss[name] = 0.5*ADMM.rhos[name]*(torch.norm(W-ADMM.alpha[name]*ADMM.ADMM_Q[name]+ADMM.ADMM_U[name],p=2)**2)
        mixed_loss = 0
        mixed_loss += ce_loss
        for k,v in admm_loss.items():
             mixed_loss+=v
        return ce_loss,admm_loss,mixed_loss



def admm_adjust_learning_rate(optimizer,epoch,args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = args.admm_epoch
    lr = None
    if epoch % admm_epoch == 0:
         lr = args.lr
    else:
         admm_epoch_offset = epoch%admm_epoch

         admm_step = admm_epoch/3  # roughly every 1/3 admm_epoch. 
         
         lr = args.lr *(0.1 ** (admm_epoch_offset//admm_step)) 

    for param_group in optimizer.param_groups:
         param_group['lr'] = lr

def zero_masking(args,config,model):
    masks = {}
    for name,W in model.named_parameters():  ## no gradient for weights that are already zero (for progressive pruning and sequential pruning)        
        if name in config.prune_ratios:            
            w_temp = W.cpu().detach().numpy()
            indices = (w_temp != 0)
            indices = indices.astype(np.float32)            
            masks[name] = torch.from_numpy(indices).cuda()
    config.zero_masks = masks
def masking(args,config,model):
    masks = {}
    for name,W in model.named_parameters():
        if name in config.prune_ratios:           
            above_threshold, pruned_weight = weight_pruning(args,W,config.prune_ratios[name])
            W.data = pruned_weight
            masks[name] = above_threshold
            
    config.masks = masks


    

def log_weight_histogram(model,epoch):
     """
     log convergence condition, admm loss and weight histogram to tensorboard to view
     Args:
       model: model to extract weights

       epoch: x axis
     """
     for tag,value in model.named_parameters():
          tag = tag.replace('.','/')
          logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
          logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
def log_loss(tag,value,epoch):

    logger.scalar_summary(tag, value.item(), epoch)
