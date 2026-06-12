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

## Usage

```
$ uv sync
$ uv run main.py
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
