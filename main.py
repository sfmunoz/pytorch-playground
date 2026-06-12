#!/usr/bin/env python3
# vim: set foldmethod=marker:

# Ref: PyTorch in 1 Hour -> https://www.youtube.com/watch?v=r1bquDz5GGA

# {{{ imports

import sys
import torch
import torch.nn as nn
import torch.optim as optim
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
# {{{ Operators.argmax()

    def argmax(self):
        log.info("==== Operators.argmax() ====")
        x = torch.tensor([[10,0,5,20,1],[1,30,2,5,0]])
        m0 = torch.argmax(x,dim=0)
        m1 = torch.argmax(x,dim=1)
        tensor_log(x," <x> ")
        tensor_log(m0,"<m0> ")
        tensor_log(m1,"<m1> ")

# }}}
# {{{ Operators.gather()

    def gather(self):
        log.info("==== Operators.gather() ====")
        x = torch.tensor([[10,11,12,13],[20,21,22,23],[30,31,32,33]])
        i = torch.tensor([[2],[0],[3]])
        v = torch.gather(x,dim=1,index=i)
        tensor_log(x,"<x> ")
        tensor_log(i,"<i> ")
        tensor_log(v,"<v> ")

# }}}
# {{{ Operators.run()

    def run(self):
        self.star_vs_at()
        self.reduction()
        self.indexing()
        self.argmax()
        self.gather()

# }}}
# -------- ModelScratch(object) -- class --------
# {{{ ModelScratch -- class

class ModelScratch(object):

# }}}
# {{{ ModelScratch.__init__()

    def __init__(self,args):
        log.info("==== ModelScratch.__init__() ====")
        self.__args = args
        self.__n = 10
        self.__d_in = 1
        self.__d_out = 1
        self.__lr = 0.01
        self.__epochs = 200
        self.__noise = 0.1
        self.__x = torch.randn(self.__n,self.__d_in)
        self.__w_true = torch.tensor([[2.0]])
        self.__b_true = torch.tensor([[1.0]])
        self.__y_true = self.__x @ self.__w_true + self.__b_true + torch.randn(self.__n,self.__d_out) * self.__noise
        self.__w = torch.randn(self.__d_in,self.__d_out,requires_grad=True)
        self.__b = torch.randn(self.__d_in,self.__d_out,requires_grad=True)
        log.info(f"n={self.__n} | d_in={self.__d_in} | d_out={self.__d_out} | lr={self.__lr} | epochs={self.__epochs}")
        tensor_log(self.__x,"     <x> ")
        tensor_log(self.__w_true,"<w_true> ")
        tensor_log(self.__b_true,"<b_true> ")
        tensor_log(self.__y_true,"<y_true> ")
        tensor_log(self.__w,"     <w> ")
        tensor_log(self.__b,"     <b> ")

# }}}
# {{{ ModelScratch.run()

    def run(self):
        log.info("==== ModelScratch.run() ====")
        for epoch in range(1,self.__epochs+1):
            y_hat = self.__x @ self.__w + self.__b
            err = y_hat - self.__y_true
            sq_err = err ** 2
            loss = sq_err.mean()
            loss.backward()  # calcs ".grad"
            with torch.no_grad():
                self.__w -= self.__lr * self.__w.grad
                self.__b -= self.__lr * self.__b.grad
            self.__w.grad.zero_()
            self.__b.grad.zero_()
            if epoch % 5 != 0:
                continue
            log.info(f"epoch={epoch:03d} | loss={loss.item():.4f} | w={self.__w.item()} | b={self.__b.item()}")
        log.info(f"final: w={self.__w.item()} | b={self.__b.item()}")
        log.info(f" true: w={self.__w_true.item()} | b={self.__b_true.item()}")

# }}}
# -------- ModelNN(nn.Module) -- class --------
# {{{ ModelNN -- class

class ModelNN(nn.Module):

# }}}
# {{{ ModelNN.__init__()

    def __init__(self,args):
        log.info("==== ModelNN.__init__() ====")
        super().__init__()
        self.__args = args
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
        tensor_log(self.linear_layer.weight,"<ll.weight> ")
        tensor_log(self.linear_layer.bias,"  <ll.bias> ")
        for x in str(self).split("\n"):
              log.info("    <model> " + x)
        for p in self.parameters():
            for x in str(p).split("\n"):
                log.info("    <param> " + x)
        self.__lr = 0.01
        self.__optimizer = optim.Adam(self.parameters(),lr=self.__lr)
        self.__loss_fn = nn.MSELoss()
        self.__n = 10
        self.__d_in = 1
        self.__d_out = 1
        self.__epochs = 200
        self.__noise = 0.1
        self.__x = torch.randn(self.__n,self.__d_in)
        self.__w_true = torch.tensor([[2.0]])
        self.__b_true = torch.tensor([[1.0]])
        self.__y_true = self.__x @ self.__w_true + self.__b_true + torch.randn(self.__n,self.__d_out) * self.__noise
        log.info(f"n={self.__n} | d_in={self.__d_in} | d_out={self.__d_out} | lr={self.__lr} | epochs={self.__epochs}")
        tensor_log(self.__x,"     <x> ")
        tensor_log(self.__w_true,"<w_true> ")
        tensor_log(self.__b_true,"<b_true> ")
        tensor_log(self.__y_true,"<y_true> ")
        tensor_log(self.linear_layer.weight,"     <w> ")
        tensor_log(self.linear_layer.bias,"     <b> ")

# }}}
# {{{ ModelNN.forward()

    def forward(self,x):
        return self.linear_layer(x)

# }}}
# {{{ ModelNN.run()

    def run(self):
        log.info("==== ModelNN.run() ====")
        for epoch in range(1,self.__epochs+1):
            y_hat = self(self.__x)
            loss = self.__loss_fn(y_hat,self.__y_true)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            if epoch % 5 != 0:
                continue
            log.info(f"epoch={epoch:03d} | loss={loss.item():.4f} | w={self.linear_layer.weight.item()} | b={self.linear_layer.bias.item()}")
        log.info(f"final: w={self.linear_layer.weight.item()} | b={self.linear_layer.bias.item()}")
        log.info(f" true: w={self.__w_true.item()} | b={self.__b_true.item()}")

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
    ModelScratch(args).run()
    ModelNN(args).run()

# }}}
