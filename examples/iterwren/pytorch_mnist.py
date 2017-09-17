import numpy as np
import time
import pywren
import logging
import threading
import pywrenext.iterwren
import pywrenext.sdblogger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

import pickle

import daiquiri
 
LOG_FILENAME = "pytorch_mnist.iterwren.log"
try:
    os.remove(LOG_FILENAME)
except OSError:
    pass


daiquiri.setup( outputs=[daiquiri.output.File(LOG_FILENAME)])

iw = logging.getLogger('pywrenext.iterwren.iterwren')
iw.setLevel('DEBUG')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



BATCH_SIZE = 32

def ser(model, optimizer):
    model_state = model.cpu().state_dict()
    optimizer_state = optimizer.state_dict()
    return pickle.dumps({'model_state' :  model_state, 
                         'optimizer_state' : optimizer_state}, -1)

def deser(s):
    return pickle.loads(s)

def pt_iter(k, x_k, args):

    import torch

    timelog = {}
    timelog['start_iter'] = time.time()
    sdblog = args['sdblog']
    USE_CUDA = args['use_gpu']
    kwargs = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)


    USE_SDB_LOGGER = True
    # generic setup
    model = Net()
    opt_lr = args['learning_rate']
    opt_momentum = args.get('opt_momentum', 0.5)
    
    optimizer = optim.SGD(model.parameters(), lr=opt_lr, 
                          momentum=opt_momentum)

    # just return initial state
    if k == 0:
        return {'model_state' : ser(model, optimizer)}

    if USE_CUDA:
        model = model.cuda()

    x_k = deser(x_k['model_state']) # convert from string back to dict

    model.load_state_dict(x_k['model_state'])
    if k > 1:
        optimizer.load_state_dict(x_k['optimizer_state'])

    timelog['setup_complete'] = time.time()

    # do a train epoch

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
                # sdblog(loss=loss.data[0],
                #        batch_idx=batch_idx,
                #        epoch=epoch, 
                #        field='epoch_loss')


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),accuracy))
        return accuracy


    train(k)
    timelog['train_epoch_complete'] = time.time()
    test_accuracy = test()
    timelog['test_complete'] = time.time()
    sdblog(k=k, test_accuracy=test_accuracy, field='test_loss', 
           arg_i = args['arg_i'])
    
    model_state = ser(model, optimizer)
    timelog['ser_complete'] = time.time()
    state = {'model_state' : model_state, 
             'test_accuracy' : test_accuracy, 
             'timelog' : timelog}

    return state

config = pywren.wrenconfig.default()
config['runtime']['s3_bucket'] = 'pywren-public-us-west-2'
config['runtime']['s3_key'] = 'pywren.runtimes/deep_gpu_3.6.meta.json'


sdblog = pywrenext.sdblogger.SDBLogger('jonas-cnn-log')
print("SDB LOG IS", sdblog)

NUM_EPOCHS = 10

config = pywren.wrenconfig.load("pywrenconfig_gpu.yaml")
wrenexec = pywren.standalone_executor(config=config)

args = [{'learning_rate' : 0.01, 
         'opt_momentum' : 0.5, 
         'sdblog' : sdblog, 
         'use_gpu' : ug, 
         'arg_i' : i } for i, ug in enumerate([False, True, False, True])]

with pywrenext.iterwren.IterExec(wrenexec) as IE:

    iter_futures = IE.map(pt_iter, NUM_EPOCHS, args, 
                          save_iters=True)
    pywrenext.iterwren.wait_exec(IE)

print(iter_futures[0].current_future)

iter_futures_hist = [f.iter_hist for f in iter_futures]
current_futures = [f.current_future for f in iter_futures]

for f in iter_futures_hist[0]:
    print(f.result().get('test_accuracy', 0))

pickle.dump({'iter_futures_hist' : iter_futures_hist, 
             'current_futures' : current_futures, 
             'args' : args}, 
            open("pytorch_mnist.pickle", 'wb'), -1)
print("results dumped")

