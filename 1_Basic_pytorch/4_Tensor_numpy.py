# torch <-> numpy
    # transformed the between Tensor and Array
    # numpy()
    # from_numpy()
    # ** if Tensor in CPU -> Numpy Array is shared the Memory -> one change other change too **

import torch
import numpy as np

# transformed the tensor to numpy
a = torch.ones(7)
b = a.numpy()
print(a, a.dtype)
print(b, b.dtype)

# Memory share
a.add_(1)       
print(a)        # tensor([2., 2., 2., 2., 2., 2., 2.])
print(b)        # [2. 2. 2. 2. 2. 2. 2.]

# transformed the numpy to tensor
a = np.ones(7)          
b = torch.from_numpy(b)
np.add(a, 1, out=a)
print(a)
print(b)

# cuda tensor : move the tensor on any device(cpu, gpu)
x = torch.randn(1)
print(x)        # tensor([-0.1575])
print(x.item()) # -0.1574528068304062
print(x.dtype)  # torch.float32

device = "cuda" if torch.cuda.is_available() else "cpu"
y = torch.ones_like(x, device=device)
x = x.to(device=device)
z = torch.add(x,y)
print(device)               # cuda
print(z)                    # tensor([1.8775], device='cuda:0')
print(z.to(device="cpu"))   # tensor([1.8775])
