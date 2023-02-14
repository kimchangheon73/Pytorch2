# torchvision
    # transforms : 전처리 할때 사용하는 메서드
    # https://pytorch.org/docs/stable/torchvision/transforms.html

import torch
import torchvision
import torchvision.transforms as trasnforms
transform = trasnforms.Compose([trasnforms.ToTensor(),
                                trasnforms.Normalize(mean=(0.5), std=(0.5))])


# utils.data
    # 다양한 샘플 데이터 존재
    # DataLoader, Dataset을 통해 Batch_size, train여부, transform등을 넣어 load 방법을 지정

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as trans

# download the data
trainset = torchvision.datasets.MNIST(root="./data/", train=True, download = True, transform = trans.ToTensor())
testset = torchvision.datasets.MNIST(root="./data/", train=False, download = True, transform = trans.ToTensor())

# Loading the data
train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

# Extract one data
dataiter = iter(train_loader)       
img, label = dataiter._next_data()
print(img.shape)        # torch.Size([8, 1, 28, 28]) --> torch에서는 채널이 앞에 옴 
print(label.shape)      # torch.Size([8])

# Check the data
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
torch_img = torch.squeeze(img[0])       # torch.Size([28, 28])
numpy_img = torch_img.numpy()           # (28, 28)
label = torch.squeeze(label[0]).item()  # label

plt.title(label)
plt.imshow(numpy_img, cmap="gray")
plt.axis("off")
plt.show()
