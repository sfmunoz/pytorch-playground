#!/usr/bin/env python3
# vim: set foldmethod=marker:

# Refs:
# - PyTorch Course (2022), Part 4: Image Classification (MNIST) -> https://www.youtube.com/watch?v=gBw0u_5u0qU 
# - PyTorch Project: Handwritten Digit Recognition -> https://www.youtube.com/watch?v=vBlO87ZAiiw

# {{{ imports

import sys, os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from logging import getLogger, basicConfig, INFO, DEBUG
basicConfig(format='%(asctime)s [%(relativeCreated)7.0f] [%(levelname).1s] %(message)s',level=INFO,stream=sys.stderr)
log = getLogger(__name__)

# }}}
# {{{ globals

# torch.cuda.is_available()
# torch.cuda.device_count()
# torch.cuda.get_device_name(0)
# torch.cuda.device(0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(DEVICE)

# }}}
# -------- functions --------
# {{{ tensor_log()

def tensor_log(t,p="<tensor> ",d=False,f=log.info):
    f(p + f"shape={t.shape} | dtype={t.dtype} | device={t.device} | requires_grad={t.requires_grad} | grad_fn={t.grad_fn}")
    if not d:
        return
    for x in str(t).split("\n"):
        f(p + x)

# }}}
# {{{ model_log()

def model_log(m,pm=" <model> ",pp=" <param> ",d=False,f=log.info):
    for x in str(m).split("\n"):
        f(pm + x)
    if not d:
        return
    for p in m.parameters():
        for x in str(p).split("\n"):
            f(pp + x)

# }}}
# {{{ dataset_log()

def dataset_log(d,p="<dataset> ",f=log.info):
    for x in str(d).split("\n"):
        f(p + x)

# }}}
# {{{ img_plot()

SHADES = " .:-=+*#%@"
N_SHADES = len(SHADES)

def img_plot(img, p="<img> ", f=log.info):
    if img.dim() != 2:
        raise Exception(f"expected 2D tensor (HxW) got {img.dim()}D")
    if img.shape[0] != 28 or img.shape[1] != 28:
        raise Exception(f"expected 28x28 got {list(img.shape)}")
    v_min = img.min().item()
    v_max = img.max().item()
    span = v_max - v_min if v_max != v_min else 1.0
    for row in img:
        line = "".join(SHADES[int(((v - v_min) / span) * (N_SHADES - 1))] for v in row.tolist())
        f(p + line)

# }}}
# -------- MyNet(nn.Module) -- class --------
# {{{ MyNet -- class

class MyNet(nn.Module):

# }}}
# {{{ MyNet.__init__()

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # <batch-size> x 1 x 28 x 28 -> <batch-size> x 784
        self.fc1 = nn.Linear(28**2,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,10)
        self.R = nn.ReLU()

# }}}
# {{{ MyNet.forward()

    def forward(self,x):
        x = self.flatten(x)
        x = self.R(self.fc1(x))
        x = self.R(self.fc2(x))
        return self.fc3(x)

# }}}
# -------- Mnist(object) -- class --------
# {{{ Mnist -- class

class Mnist(object):

# }}}
# {{{ Mnist.__init__()

    def __init__(self,args):
        log.info("==== Mnist.__init__() ====")
        self.__args = args
        self.__mnist_mean = 0.1307
        self.__mnist_std = 0.3081
        if os.getenv("MNIST_CALC") == "1":
            raw_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
            all_images = torch.stack([img for img, _ in self.__train_data], dim=0)
            log.info(f"all_images ... {all_images.shape}")  # 60000,1,28,28
            #tensor_log(all_images,"<all_images> ")  # 60000,1,28,28
            self.__mnist_mean = all_images.mean().item()  # 0.1307
            self.__mnist_std = all_images.std().item()    # 0.3081
        log.info(f"mnist_mean ... {self.__mnist_mean:.4f}")
        log.info(f"mnist_std .... {self.__mnist_std:.4f}")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.__mnist_mean, self.__mnist_std),
        ])
        self.__train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.__test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        dataset_log(self.__train_data,"<train_data> ")
        dataset_log(self.__test_data," <test_data> ")
        log.info(f"train_data ... {len(self.__train_data)} samples")  # 60000
        log.info(f"test_data .... {len(self.__test_data)} samples")   # 10000
        if log.isEnabledFor(DEBUG):
            for i in range(3):
                img_plot(self.__train_data[i][0].squeeze(),f"<{self.__train_data[i][1]}> ",f=log.debug)  # squeeze(): A×1×B×C×1×D -> A×B×C×D
        self.__batch_size = 25
        self.__train_loader = DataLoader(self.__train_data,batch_size=self.__batch_size,shuffle=True,num_workers=1)
        self.__test_loader = DataLoader(self.__test_data,batch_size=self.__batch_size,shuffle=True,num_workers=1)
        log.info(f"train_loader ... {self.__batch_size:3d} x {len(self.__train_loader):5d} = {len(self.__train_data):5d}")
        log.info(f"test_loader .... {self.__batch_size:3d} x {len(self.__test_loader):5d} = {len(self.__test_data):5d}")
        self.__model = MyNet()
        model_log(self.__model)

# }}}
# {{{ Mnist.__run_train()

    def __run_train(self):
        log.info("==== Mnist.__run__train() ====")
        n_epochs = self.__args.epochs
        n_batches = len(self.__train_loader)
        log.info(f"epochs ....... {n_epochs:5d}")
        log.info(f"batch_size ... {self.__batch_size:5d}")
        log.info(f"batches ...... {n_batches:5d}")
        log.info(f"samples ...... {self.__batch_size * n_batches:5d}")
        optimizer = optim.Adam(self.__model.parameters(),lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        self.__model.train()
        for epoch in range(1, n_epochs + 1):
            log.info(f"#### epoch {epoch}/{n_epochs} ####")
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
            for i,(data,target) in enumerate(self.__train_loader,1):
                #tensor_log(data,"  <data> ")    # <batch-size> x 1 x 28 x 28
                #tensor_log(target,"<target> ")  # <batch-size>
                y_hat = self.__model(data)
                loss = loss_fn(y_hat, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, predicted = y_hat.max(1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                if i % 100 == 0:
                    log.info(f"  epoch {epoch:2d}/{n_epochs} | batch {i:5d}/{n_batches} | train_loss {epoch_loss/i:.4f}")
            avg_loss = epoch_loss / n_batches
            acc = 100.0 * epoch_correct / epoch_total
            log.info(f"==== epoch {epoch:2d}/{n_epochs} done | train_loss {avg_loss:.4f} | train_acc {acc:.2f}% ====")

# }}}
# {{{ Mnist.__run_test()

    def __run_test(self):
        log.info("==== Mnist.__run_test() ====")
        n_batches = len(self.__test_loader)
        log.info(f"batch_size ... {self.__batch_size:5d}")
        log.info(f"batches ...... {n_batches:5d}")
        log.info(f"samples ...... {self.__batch_size * n_batches:5d}")
        loss_fn = nn.CrossEntropyLoss()
        self.__model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for i,(data,target) in enumerate(self.__test_loader,1):
                y_hat = self.__model(data)
                loss = loss_fn(y_hat, target)
                total_loss += loss.item()
                _, predicted = y_hat.max(1)
                total_samples += target.size(0)
                total_correct += (predicted == target).sum().item()
                if i % 100 == 0:
                    log.info(f"  test batch {i:5d}/{n_batches} | test_loss {total_loss/i:.4f}")
        avg_loss = total_loss / n_batches
        acc = 100.0 * total_correct / total_samples
        log.info(f"==== test done | test_loss {avg_loss:.4f} | test_acc {acc:.2f}% ====")

# }}}
# {{{ Mnist.run()

    def run(self):
        log.info("==== Mnist.run() ====")
        self.__run_train()
        self.__run_test()

# }}}
# -------- main --------
# {{{ main

if __name__ == "__main__":
    parser = ArgumentParser(
        description = 'mnist.py (v1.0)',
        epilog = "sfmunoz (C) 2026",
    )

    parser.add_argument('-d', '--debug', action='store_true',
                        help='enable debug mode')

    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='number of training epochs (default: 1)')

    args = parser.parse_args()

    if args.debug:
        log.setLevel(DEBUG)

    Mnist(args).run()

# }}}
