import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import numpy as np
import json
import random

import sys
import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
        
def to_np():
    to_np = lambda x : x.data.cpu().numpy()    
    return to_np

def concat():
    concat = lambda x : np.concatenate(x, axis=0)
    return concat

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, local_rank=0, no_save=False):
        self.terminal = sys.stdout
        self.file = None
        self.local_rank = local_rank
        self.no_save = no_save
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        if self.local_rank and not self.no_save == 0: self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if self.local_rank == 0:
            if '\r' in msg: is_file = 0
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1 and not self.no_save:
                self.file.write(msg)
                self.file.flush()
    def flush(self): 
        pass

def sampler(args, train_dataset):
    
    if args.benchmark == 'cifar10':
        classes = copy.deepcopy(train_dataset.classes())
        d = {}
        for i in range(10):
            d[i] = list(np.where(np.array(train_dataset.classes())==i)[0])

        batch = []
        for _ in range(len(classes)//args.bs):
            tmp = []
            if args.collapse:
                cls = np.random.choice(list(range(10)), args.bs, replace=True)
            else:
                cls = np.random.choice(list(range(10)), args.bs, replace=False)
            for i in cls:
                tmp.append(int(np.random.choice(d[i], 1)))
            batch.append(tmp)
    elif args.benchmark == 'cifar100':
        classes = copy.deepcopy(train_dataset.classes())
        d = {}
        for i in range(100):
            d[i] = list(np.where(np.array(train_dataset.classes())==i)[0])

        batch = []
        for _ in range(len(classes)//args.bs):
            tmp = []
            if args.collapse:
                cls = np.random.choice(list(range(100)), args.bs, replace=True)
            else:
                cls = np.random.choice(list(range(100)), args.bs, replace=False)
            for i in cls:
                tmp.append(int(np.random.choice(d[i], 1)))
            batch.append(tmp)
            
    elif args.benchmark == 'imagenet200':
        classes = copy.deepcopy(train_dataset.classes())
        d = {}
        for i in range(200):
            d[i] = list(np.where(np.array(train_dataset.classes())==i)[0])

        batch = []
        for _ in range(len(classes)//args.bs):
            tmp = []
            if args.collapse:
                cls = np.random.choice(list(range(200)), args.bs, replace=True)
            else:
                cls = np.random.choice(list(range(200)), args.bs, replace=False)
            for i in cls:
                tmp.append(int(np.random.choice(d[i], 1)))
            batch.append(tmp)
    elif args.benchmark == 'imagenet':
        classes = copy.deepcopy(train_dataset.classes())
        d = {}
        for i in range(1000):
            d[i] = list(np.where(np.array(train_dataset.classes())==i)[0])

        batch = []
        for _ in range(len(classes)//args.bs):
            tmp = []
            if args.collapse:
                cls = np.random.choice(list(range(1000)), args.bs, replace=True)
            else:
                cls = np.random.choice(list(range(1000)), args.bs, replace=False)
            for i in cls:
                tmp.append(int(np.random.choice(d[i], 1)))
            batch.append(tmp)

    return batch