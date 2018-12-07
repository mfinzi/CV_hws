import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
from random import choice
import utils


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, bias=False)
        self.S2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        self.S4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.S2(x)
        x = F.relu(self.conv2(x))
        x = self.S4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        return x

def train_batch(x, y, clf, opt, args):
    """Training step for one single batch (one forward, one backward, and one update)
    Args:
        [x]     FloatTensor (B, C, W, H), Images
        [y]     LongTensor  (B, 1), labels
        [clf]   Classifier model
        [opt]   Optimizer that updates the weights for the classifier.
        [args]  Commandline arguments.
    Rets:
        [loss]      (float) Loss of the batch (before update).
        [ncorrect]  (float) Number of correct examples in the batch (before update).
    """
    opt.zero_grad()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        x = x.cuda()
        y = y.cuda()

    res = clf(x)
    ncorrect = 0
    for i in range(y.size(0)):
        _, ind = res[i].max(0)

        if torch.equal(y[i], ind):
            ncorrect +=1

    loss = criterion(res, y)
    loss.backward()
    opt.step()

    return loss, ncorrect


def evaluate(clf, loader, args):
    """Evaluate the classfier on the dataset loaded by the [loader]
    Args:
        [clf]       Model to be evaluated.
        [loader]    Dataloader, which will provide the test-set.
        [args]      Commandline arguments.
    Rets:
        Dictionary with following structure: {
            'loss'  : test loss (averaged across samples),
            'acc'   : test accuracy
        }
    """
    
    res_dict = {
    'loss' : 0,
    'acc' : 0
    }
    clf.eval()

    criterion = nn.CrossEntropyLoss()
    ncorrect = 0.
    n = 0.
    for x, y in loader:
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        n += float(x.size(0))

        outputs = clf(x)

        for i in range(y.size(0)):
            _, ind = outputs[i].max(0)

            if torch.equal(y[i], ind):
                ncorrect +=1

        loss = criterion(outputs, y)

        res_dict['loss'] += loss

    res_dict['loss'] = res_dict['loss']/n
    res_dict['loss'] = res_dict['loss'][0].item()
    res_dict['acc'] = ncorrect/n
    clf.train()
    return res_dict

def resume_model(filename, args):
    """Resume the training (both model and optimizer) with the checkpoint.
    Args:
        [filename]  Str, file name of the checkpoint.
        [args]      Commandline arguments.
    Rets:
        [clf]   CNN with weights loaded with the pretrained weights from checkpoint [filename]
        [opt]   Optimizer with parameters resumed from the checkpoint [filename]
    """
    checkpoint = torch.load(filename)
    clf = CNN()
    clf.load_state_dict(checkpoint['clf'])
    
    if args.cuda:
        clf = clf.cuda()

    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    opt.load_state_dict(checkpoint['opt'])

    return clf, opt


############################################################
# DO NOT MODIFY CODES BELOW
############################################################
if __name__ == "__main__":
    args = utils.get_args()
    loaders = utils.get_loader(args)
    writer = SummaryWriter()

    clf = CNN()
    if args.cuda:
        clf = clf.cuda()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    if args.resume is not None:
        clf, opt = resume_model(args.resume, args)

    step = 0
    for epoch in range(20):
        print("Epoch:%d"%(epoch+1))
        for x, y in loaders['train']:
            l, c = train_batch(x, y, clf, opt, args)
            acc = c/float(x.size(0))
            step += 1
            if step % 250 == 0:
                print("Step:%d\tLoss:%2.5f\tAcc:%2.5f"%(step, l, acc))

        print("Evaluating testing error:")
        res = evaluate(clf, loaders['test'], args)
        print(res)

    utils.save_checkpoint('clf.pth.tar', **{
        'clf' : clf.state_dict(),
        'opt' : opt.state_dict()
    })

