# pytorch-playground

PyTorch playground

## References

- https://pytorch.org/
- https://github.com/pytorch/pytorch
- https://en.wikipedia.org/wiki/PyTorch

### Videos

- [PyTorch in 1 Hour](https://www.youtube.com/watch?v=r1bquDz5GGA)
- [PyTorch Crash Course - Getting Started with Deep Learning](https://www.youtube.com/watch?v=OIenNRt2bjg)
- [PyTorch Crash Course: Deep Learning in Python](https://www.youtube.com/watch?v=uq7sbUlIDR8)
- [Visual Introduction to PyTorch](https://www.youtube.com/watch?v=G4UAQ6bxQzE)

## Install

**uv.lock** is intentionally left out provided that this is a playground project:

```
$ uv sync                                                            # CUDA 13.0 (https://pypi.org/simple/)
$ uv sync --default-index https://download.pytorch.org/whl/cpu       # CPU
$ uv sync --default-index https://download.pytorch.org/whl/cu130     # CUDA 13.0
$ uv sync --default-index https://download.pytorch.org/whl/cu132     # CUDA 13.2
$ uv sync --default-index https://download.pytorch.org/whl/xpu       # XPU
```

## Run

```
$ uv run main.py
$ uv run mnist.py
```

## The 5-step deep learning recipe

From [PyTorch in 1 Hour → 1m26s](https://www.youtube.com/watch?v=r1bquDz5GGA&t=86s):

| Step | Concept (Math) | From Scratch (Raw Tensors) | Professional (`torch.nn`) |
| --- | --- | --- | --- |
| **1. Prediction** | Make a prediction: $$\hat{y} = f(X, \theta)$$ | `y_hat = X @ W + b` | `y_hat = model(X)` |
| **2. Loss Calc** | Quantify the error: $$L = \text{Loss}(\hat{y}, y)$$ | `loss = torch.mean((y_hat - y)**2)` | `loss = criterion(y_hat, y)` |
| **3. Gradient Calc** | Find the slope of the loss: $$\nabla_{\theta}L$$ | `loss.backward()` | `loss.backward()` |
| **4. Param Update** | Step down the slope: $$\theta_{t+1} = \theta_t - \eta\nabla_{\theta}L$$ | `W -= lr * W.grad` | `optimizer.step()` |
| **5. Gradient Reset** | Reset for the next loop. | `W.grad.zero_()` | `optimizer.zero_grad()` |

## mnist.py execution example

One training epoch, using CPU:

```
$ uv run mnist.py
2026-06-14 09:35:57,630 [   3787] [I] ==== Mnist.__init__() ====
2026-06-14 09:35:57,630 [   3787] [I] mnist_mean ... 0.1307
2026-06-14 09:35:57,630 [   3787] [I] mnist_std .... 0.3081
2026-06-14 09:35:57,864 [   4021] [I] <train_data> Dataset MNIST
2026-06-14 09:35:57,865 [   4021] [I] <train_data>     Number of datapoints: 60000
2026-06-14 09:35:57,865 [   4021] [I] <train_data>     Root location: ./data
2026-06-14 09:35:57,865 [   4021] [I] <train_data>     Split: Train
2026-06-14 09:35:57,865 [   4021] [I] <train_data>     StandardTransform
2026-06-14 09:35:57,865 [   4021] [I] <train_data> Transform: Compose(
2026-06-14 09:35:57,865 [   4021] [I] <train_data>                ToTensor()
2026-06-14 09:35:57,865 [   4022] [I] <train_data>                Normalize(mean=0.1307, std=0.3081)
2026-06-14 09:35:57,865 [   4022] [I] <train_data>            )
2026-06-14 09:35:57,865 [   4022] [I]  <test_data> Dataset MNIST
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>     Number of datapoints: 10000
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>     Root location: ./data
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>     Split: Test
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>     StandardTransform
2026-06-14 09:35:57,865 [   4022] [I]  <test_data> Transform: Compose(
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>                ToTensor()
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>                Normalize(mean=0.1307, std=0.3081)
2026-06-14 09:35:57,865 [   4022] [I]  <test_data>            )
2026-06-14 09:35:57,865 [   4022] [I] train_data ... 60000 samples
2026-06-14 09:35:57,865 [   4022] [I] test_data .... 10000 samples
2026-06-14 09:35:57,866 [   4022] [I] train_loader ...  25 x  2400 = 60000
2026-06-14 09:35:57,866 [   4023] [I] test_loader ....  25 x   400 = 10000
2026-06-14 09:35:57,867 [   4024] [I]  <model> MyNet(
2026-06-14 09:35:57,867 [   4024] [I]  <model>   (flatten): Flatten(start_dim=1, end_dim=-1)
2026-06-14 09:35:57,867 [   4024] [I]  <model>   (fc1): Linear(in_features=784, out_features=16, bias=True)
2026-06-14 09:35:57,867 [   4024] [I]  <model>   (fc2): Linear(in_features=16, out_features=16, bias=True)
2026-06-14 09:35:57,867 [   4024] [I]  <model>   (fc3): Linear(in_features=16, out_features=10, bias=True)
2026-06-14 09:35:57,867 [   4024] [I]  <model>   (R): ReLU()
2026-06-14 09:35:57,867 [   4024] [I]  <model> )
2026-06-14 09:35:57,867 [   4024] [I] ==== Mnist.run() ====
2026-06-14 09:35:57,867 [   4024] [I] ==== Mnist.__run__train() ====
2026-06-14 09:35:57,868 [   4024] [I] epochs .......     1
2026-06-14 09:35:57,868 [   4025] [I] batch_size ...    25
2026-06-14 09:35:57,868 [   4025] [I] batches ......  2400
2026-06-14 09:35:57,868 [   4025] [I] samples ...... 60000
2026-06-14 09:35:57,868 [   4025] [I] #### epoch 1/1 ####
2026-06-14 09:36:03,347 [   9504] [I]   epoch  1/1 | batch   100/2400 | train_loss 1.6448
2026-06-14 09:36:05,245 [  11402] [I]   epoch  1/1 | batch   200/2400 | train_loss 1.2133
2026-06-14 09:36:07,072 [  13229] [I]   epoch  1/1 | batch   300/2400 | train_loss 0.9865
2026-06-14 09:36:08,830 [  14986] [I]   epoch  1/1 | batch   400/2400 | train_loss 0.8582
2026-06-14 09:36:10,623 [  16779] [I]   epoch  1/1 | batch   500/2400 | train_loss 0.7683
2026-06-14 09:36:12,256 [  18413] [I]   epoch  1/1 | batch   600/2400 | train_loss 0.7097
2026-06-14 09:36:13,815 [  19972] [I]   epoch  1/1 | batch   700/2400 | train_loss 0.6583
2026-06-14 09:36:15,552 [  21709] [I]   epoch  1/1 | batch   800/2400 | train_loss 0.6211
2026-06-14 09:36:17,346 [  23503] [I]   epoch  1/1 | batch   900/2400 | train_loss 0.5883
2026-06-14 09:36:18,858 [  25015] [I]   epoch  1/1 | batch  1000/2400 | train_loss 0.5617
2026-06-14 09:36:20,637 [  26794] [I]   epoch  1/1 | batch  1100/2400 | train_loss 0.5409
2026-06-14 09:36:22,131 [  28288] [I]   epoch  1/1 | batch  1200/2400 | train_loss 0.5217
2026-06-14 09:36:23,712 [  29868] [I]   epoch  1/1 | batch  1300/2400 | train_loss 0.5072
2026-06-14 09:36:25,443 [  31599] [I]   epoch  1/1 | batch  1400/2400 | train_loss 0.4917
2026-06-14 09:36:26,700 [  32857] [I]   epoch  1/1 | batch  1500/2400 | train_loss 0.4812
2026-06-14 09:36:27,974 [  34131] [I]   epoch  1/1 | batch  1600/2400 | train_loss 0.4684
2026-06-14 09:36:29,520 [  35676] [I]   epoch  1/1 | batch  1700/2400 | train_loss 0.4577
2026-06-14 09:36:31,025 [  37182] [I]   epoch  1/1 | batch  1800/2400 | train_loss 0.4464
2026-06-14 09:36:32,582 [  38739] [I]   epoch  1/1 | batch  1900/2400 | train_loss 0.4384
2026-06-14 09:36:34,561 [  40718] [I]   epoch  1/1 | batch  2000/2400 | train_loss 0.4301
2026-06-14 09:36:36,113 [  42269] [I]   epoch  1/1 | batch  2100/2400 | train_loss 0.4217
2026-06-14 09:36:37,712 [  43869] [I]   epoch  1/1 | batch  2200/2400 | train_loss 0.4152
2026-06-14 09:36:39,256 [  45412] [I]   epoch  1/1 | batch  2300/2400 | train_loss 0.4087
2026-06-14 09:36:40,809 [  46966] [I]   epoch  1/1 | batch  2400/2400 | train_loss 0.4029
2026-06-14 09:36:40,845 [  47002] [I] ==== epoch  1/1 done | train_loss 0.4029 | train_acc 88.01% ====
2026-06-14 09:36:40,845 [  47002] [I] ==== Mnist.__run_test() ====
2026-06-14 09:36:40,846 [  47002] [I] batch_size ...    25
2026-06-14 09:36:40,846 [  47002] [I] batches ......   400
2026-06-14 09:36:40,846 [  47002] [I] samples ...... 10000
2026-06-14 09:36:42,143 [  48300] [I]   test batch   100/400 | test_loss 0.2529
2026-06-14 09:36:43,238 [  49395] [I]   test batch   200/400 | test_loss 0.2575
2026-06-14 09:36:44,559 [  50716] [I]   test batch   300/400 | test_loss 0.2517
2026-06-14 09:36:45,967 [  52123] [I]   test batch   400/400 | test_loss 0.2563
2026-06-14 09:36:45,988 [  52145] [I] ==== test done | test_loss 0.2563 | test_acc 92.24% ====
```
