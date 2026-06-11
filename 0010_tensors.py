#!/usr/bin/env python3
# vim: set foldmethod=marker:

# {{{ imports

import sys
import torch
from argparse import ArgumentParser

from logging import getLogger, basicConfig, INFO
basicConfig(format='%(asctime)s [%(relativeCreated)7.0f] [%(levelname).1s] %(message)s',level=INFO,stream=sys.stderr)
log = getLogger(__name__)

# }}}
# -------- functions --------
# {{{ tensor_log()

def tensor_log(tensor,p="<tensor> ",f=log.info):
    f(p + f"shape={tensor.shape} | dtype={tensor.dtype} | device={tensor.device}")
    for x in str(tensor).split("\n"):
        f(p + x)

# }}}
# -------- TensorCreation(object) -- class --------
# {{{ TensorCreation -- class

class TensorCreation(object):

# }}}
# {{{ TensorCreation.__init__()

    def __init__(self,args):
        self.__args = args

# }}}
# {{{ TensorCreation.from_data()

    def from_data(self):
        log.info("==== TensorCreation.from_data() ====")
        data = [[1,2,3],[4,5,6]]
        my_tensor = torch.tensor(data)
        tensor_log(my_tensor,"<my_tensor> ")

# }}}
# {{{ TensorCreation.from_shape()

    def from_shape(self):
        log.info("==== TensorCreation.from_shape() ====")
        shape = (2,3)
        ones = torch.ones(shape)
        zeros = torch.zeros(shape)
        random = torch.randn(shape)
        tensor_log(ones,"  <ones> ")
        tensor_log(zeros," <zeros> ")
        tensor_log(random,"<random> ")

# }}}
# {{{ TensorCreation.from_template()

    def from_template(self):
        log.info("==== TensorCreation.from_template() ====")
        template = torch.tensor([[1,2],[3,4]])
        my_tensor = torch.rand_like(template, dtype=torch.float)
        tensor_log(template," <template> ")
        tensor_log(my_tensor,"<my_tensor> ")

# }}}
# {{{ TensorCreation.run()

    def run(self):
        self.from_data()
        self.from_shape()
        self.from_template()

# }}}
# -------- main --------
# {{{ main

if __name__ == "__main__":
    parser = ArgumentParser(
        description = '0010_tensors.py (v1.0)',
        epilog = "sfmunoz (C) 2026",
    )

    parser.add_argument('-d', '--debug', action='store_true',
                        help='enable debug mode')

    args = parser.parse_args()

    if args.debug:
        from logging import DEBUG
        log.setLevel(DEBUG)

    TensorCreation(args).run()

# }}}
