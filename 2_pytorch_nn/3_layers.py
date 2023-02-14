import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trans
import torch.nn.functional as F

# nn.Conv2d
conv = nn.Conv2d(in_channels=1,    out_channels=20,    kernel_size=5,    stride=1)
layer = nn.Conv2d(1,20,5,1).to(device=torch.device("cpu"))
print(layer)        # Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

# weight 확인
weight = layer.weight
print(weight.shape) # torch.Size([20, 1, 5, 5])

# Transform weight Tensor to Numpy
weight = weight.detach()
weight = weight.numpy()
print(weight.shape) # (20, 1, 5, 5)

# check the weight
import matplotlib.pyplot as plt
plt.imshow(weight[0,0,:,:], "jet")
plt.colorbar()
plt.show()


# data test
trainset = torchvision.datasets.MNIST(root="./data/", train=True, download = True, transform = trans.ToTensor())
testset = torchvision.datasets.MNIST(root="./data/", train=False, download = True, transform = trans.ToTensor())
train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

dataiter = iter(train_loader)
img, label = dataiter._next_data()
input_img = torch.unsqueeze(img[0], dim=0)
print(input_img.shape)          # torch.Size([1, 1, 28, 28])

output_img = layer(input_img)
output_data = output_img.data
output_data = output_data.numpy()
print(output_data.shape)        # (1, 20, 24, 24)

# check the origin, weight, featureMap
plt.figure(figsize=(5,10))
plt.subplot(1,3,1)
plt.title("input")
plt.imshow(input_img[0,0,:,:])
plt.subplot(1,3,2)
plt.title("weight")
plt.imshow(weight[0,0,:,:], "jet")
plt.subplot(1,3,3)
plt.title("featureMap")
plt.imshow(output_data[0,0,:,:])
plt.show()


# Pooling
pool = nn.MaxPool2d(2,2)
print(input_img.shape)                      # torch.Size([1, 1, 28, 28])
print(pool(input_img).shape)                # torch.Size([1, 1, 14, 14])
print(F.max_pool2d(input_img,2,2).shape)    # torch.Size([1, 1, 14, 14])

# Extract Pool to numpy
pool_arr = F.max_pool2d(input_img,2,2).numpy()
print(pool_arr.shape)

# Check the data
plt.figure(figsize=(5,15))
plt.subplot(121)
plt.title("inputs")
plt.imshow(input_img[0,0,:,:])
plt.subplot(122)
plt.title("pooling")
plt.imshow(pool_arr[0,0,:,:])
plt.show()

# Linear 
img = torch.from_numpy(input_img.numpy())
print(img.shape)            # torch.Size([1, 1, 28, 28])

flatten = nn.Flatten()
print(flatten(img).shape)   # torch.Size([1, 784])

lin = nn.Linear(784, 10)
print(lin(flatten(img)).shape)  # torch.Size([1, 10])
print(lin(flatten(img)))        # tensor([[-0.3072, -0.3231,  0.0028,  0.0930, -0.1716,  0.0050, -0.2116,  0.3256, -0.1486, -0.0858]], grad_fn=<AddmmBackward0>)

plt.figure()
plt.imshow(lin(flatten(img)).detach().numpy(), "jet")
plt.colorbar()
plt.show()

# softmax
with torch.no_grad():
    flat_img = flatten(img)
    lin = nn.Linear(784, 10)(flat_img)
    softmax = F.softmax(lin, dim=1)
    print(softmax)

# Relu
inputs = torch.randn(4, 3, 28, 28).to(device="cuda")
conv2d = nn.Conv2d(3, 20, 5, 1).to(device="cuda")
output = F.relu(conv2d(inputs))
print(inputs.shape)     # torch.Size([4, 3, 28, 28])
print(output.shape)     # torch.Size([4, 20, 24, 24])

# optimizer
    # 모델의 파라미터 업데이트
    # ex)
        # optimizer = torch.optim.Adam(model.parameter(), lr = 1e-4, weight_decay = 1e-5)
        # optimizer = torch.optim.SGD(model.parameter(), lr= 1e-3)



