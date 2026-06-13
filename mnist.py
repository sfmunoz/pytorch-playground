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

from logging import getLogger, basicConfig, INFO
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

def tensor_log(t,p="<tensor> ",f=log.info):
    f(p + f"shape={t.shape} | dtype={t.dtype} | device={t.device} | requires_grad={t.requires_grad} | grad_fn={t.grad_fn}")
    for x in str(t).split("\n"):
        f(p + x)

# }}}
# {{{ model_log()

def model_log(m,pm=" <model> ",pp=" <param> ",f=log.info):
    for x in str(m).split("\n"):
        f(pm + x)
    for p in m.parameters():
        for x in str(p).split("\n"):
            f(pp + x)

# }}}
# {{{ dataset_log()

def dataset_log(d,p="<dataset> ",f=log.info):
    for x in str(d).split("\n"):
        f(p + x)

# }}}
# -------- Mnist(object) -- class --------
# {{{ Mnist -- class

class Mnist(object):

# }}}
# {{{ Mnist.__init__()

    def __init__(self,args):
        log.info("==== Mnist.__init__() ====")
        self.__args = args
        self.__train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
        self.__test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
        dataset_log(self.__train_data,"<train_data> ")
        dataset_log(self.__test_data," <test_data> ")
        log.info(f"train_data ... {len(self.__train_data)} samples")  # 60000
        log.info(f"test_data .... {len(self.__test_data)} samples")   # 10000
        self.__mnist_mean = 0.1307
        self.__mnist_std = 0.3081
        if os.getenv("MNIST_CALC") == "1":
            all_images = torch.stack([img for img, _ in self.__train_data], dim=0)
            log.info(f"all_images ... {all_images.shape}")  # 60000,1,28,28
            #tensor_log(all_images,"<all_images> ")  # 60000,1,28,28
            self.__mnist_mean = all_images.mean().item()  # 0.1307
            self.__mnist_std = all_images.std().item()    # 0.3081
        log.info(f"mnist_mean ... {self.__mnist_mean:.4f}")
        log.info(f"mnist_std .... {self.__mnist_std:.4f}")

# }}}
# {{{ Mnist.run()

    def run(self):
        log.info("==== Mnist.run() ====")
        log.warning("to be implemented")

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

    args = parser.parse_args()

    if args.debug:
        from logging import DEBUG
        log.setLevel(DEBUG)

    Mnist(args).run()

# }}}
