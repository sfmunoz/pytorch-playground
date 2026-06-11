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

def tensor_log(t,p="<tensor> ",f=log.info):
    f(p + f"shape={t.shape} | dtype={t.dtype} | device={t.device} | requires_grad={t.requires_grad} | grad_fn={t.grad_fn}")
    for x in str(t).split("\n"):
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
# -------- Autograd(object) -- Automatic Differentiation (requires_grad=True, data -> parameter, used by loss.backward()) --------
# {{{ Autograd -- class

class Autograd(object):

# }}}
# {{{ Autograd.__init__()

    def __init__(self,args):
        self.__args = args

# }}}
# {{{ Autograd.data_vs_param()

    def data_vs_param(self):
        log.info("==== Autograd.data_vs_param() ====")
        x_data = torch.tensor([[1.,2.],[3.,4.]])
        w = torch.tensor([[1.0],[2.0]], requires_grad=True)
        tensor_log(x_data,"<x_data> ")
        tensor_log(w,"     <w> ")

# }}}
# {{{ Autograd.add_mult_graph()

    def add_mult_graph(self):
        log.info("==== Autograd.add_mult_graph() ====")
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        x = torch.tensor(4.0, requires_grad=True)
        y = a + b
        z = x * y
        tensor_log(a,"<a> ")
        tensor_log(b,"<b> ")
        tensor_log(x,"<x> ")
        tensor_log(y,"<y> ")
        tensor_log(z,"<z> ")

# }}}
# {{{ Autograd.run()

    def run(self):
        self.data_vs_param()
        self.add_mult_graph()

# }}}
# -------- Operators(object) -- class --------
# {{{ Operators -- class

class Operators(object):

# }}}
# {{{ Operators.__init__()

    def __init__(self,args):
        self.__args = args

# }}}
# {{{ Operators.star_vs_at()

    def star_vs_at(self):
        log.info("==== Operators.star_vs_at() ====")
        a = torch.tensor([[1,2],[3,4]])
        b = torch.tensor([[5,6],[7,8]])
        star = a * b
        at = a @ b
        tensor_log(a,"   <a> ")
        tensor_log(b,"   <b> ")
        tensor_log(star,"<star> ")
        tensor_log(at,"  <at> ")

# }}}
# {{{ Operators.reduction()

    def reduction(self):
        log.info("==== Operators.reduction() ====")
        scores = torch.tensor([[10.,20.,30.],[5.,10.,15.]])
        mean = scores.mean()
        mean0 = scores.mean(dim=0)
        mean1 = scores.mean(dim=1)
        tensor_log(scores,"<scores> ")
        tensor_log(mean,"  <mean> ")
        tensor_log(mean0," <mean0> ")
        tensor_log(mean1," <mean1> ")

# }}}
# {{{ Operators.indexing()

    def indexing(self):
        log.info("==== Operators.indexing() ====")
        x = torch.arange(12).reshape(3,4)
        col2 = x[:,2]
        tensor_log(x,"   <x> ")
        tensor_log(col2,"<col2> ")

# }}}
# {{{ Operators.run()

    def run(self):
        self.star_vs_at()
        self.reduction()
        self.indexing()

# }}}
# -------- main --------
# {{{ main

if __name__ == "__main__":
    parser = ArgumentParser(
        description = 'main.py (v1.0)',
        epilog = "sfmunoz (C) 2026",
    )

    parser.add_argument('-d', '--debug', action='store_true',
                        help='enable debug mode')

    args = parser.parse_args()

    if args.debug:
        from logging import DEBUG
        log.setLevel(DEBUG)

    TensorCreation(args).run()
    Autograd(args).run()
    Operators(args).run()

# }}}
