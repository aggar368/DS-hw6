import os
import pandas as pd
import numpy as np

from utils.options import args
import utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

# visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

from data import dataPreparer

import warnings, math

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)

loss_lst = []

def main():
    global loss_lst

    start_epoch = 0
    best_acc = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    # data loader
    
    loader = dataPreparer.Data(args, 
                               data_path=args.src_data_path, 
                               label_path=args.src_label_path,
                               move=args.moved_files)
    
    data_loader = loader.loader_train
    data_loader_valid = loader.loader_valid
    data_loader_test = loader.loader_test
    
    
    # Create model
    print('=> Building model...')

    # load training model
    model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)
    

    # Load pretrained weights
    if args.pretrained:
 
        ckpt = torch.load(os.path.join(checkpoint.ckpt_dir, args.source_file), map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
        
    if args.inference_only:
        inference(args, data_loader_valid, model, args.output_file)
        return

    param = [param for name, param in model.named_parameters()]
    
    optimizer = optim.Adam(param, lr = args.lr)
    #optimizer = optim.SGD(param, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma = args.lr_gamma)


    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        
        train(args, data_loader, model, optimizer, epoch)
        
        valid_acc = valid(args, data_loader_valid, model)
   
        is_best = best_acc < valid_acc
        best_acc = max(best_acc, valid_acc)
        

        state = {
            'state_dict': model.state_dict(),
            
            'optimizer': optimizer.state_dict(),
            
            'scheduler': scheduler.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
    inference(args, data_loader_test, model, args.output_file)
    
    print(f'Best acc: {best_acc:.3f}\n')

    # plot saved training loss
    epoch_ind = range(1, 51)
    plt.plot(epoch_ind, loss_lst)
    plt.show()

    # plot confusion matrix
    plot_confusion_matrix(data_loader_valid, model)



def train(args, data_loader, model, optimizer, epoch):
    global loss_lst

    losses = utils.AverageMeter()

    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()
    
    num_iterations = len(data_loader)
    
    # switch to train mode
    model.train()
        
    for i, (inputs, targets, _) in enumerate(data_loader, 1):
        
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # train
        output = model(inputs)
        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()

        ## train weights        
        losses.update(loss.item(), inputs.size(0))
        loss_lst.append(losses.val)
        
        ## evaluate
        prec1, _ = utils.accuracy(output, targets, topk = (1, 5))
        acc.update(prec1[0], inputs.size(0))

        
        if i % args.print_freq == 0:     
            print(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses,
                acc = acc))
                
      
 
def valid(args, loader_valid, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            loss = criterion(preds, targets)
        
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec1[0], inputs.size(0))

 
    print(f'Validation acc {acc.avg:.3f}\n')

    return acc.avg
    

def inference(args, loader_test, model, output_file_name):
    outputs = []
    datafiles = []
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
          
            preds = model(inputs)
    
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
            
    

    output_file = dict()
    output_file['image_name'] = datafiles
    output_file['label'] = outputs
    
    output_file = pd.DataFrame.from_dict(output_file)
    output_file.to_csv(output_file_name, index = False)


def plot_confusion_matrix(loader_valid, model):
    y_pred = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            _, output = preds.topk(1, 1, True, True)
            y_pred.extend(list(output.reshape(-1).cpu().detach().numpy()))

            _, label = targets.topk(1, 1, True, True)
            y_true.extend(list(label.reshape(-1).cpu().detach().numpy()))
    
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
    
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.show()

  

if __name__ == '__main__':
    main()

