# module import 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# Set the Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set Scaling
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,), std=(0.5,))])

# Download MNIST Set
train_dataset = torchvision.datasets.MNIST(root = "./data/", train = True,   download = True,    transform = transform)
test_dataset = torchvision.datasets.MNIST(root = "./data/", train = False,   download = True,    transform = transform)

# Load the Data
Train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
Test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# Check the Data1
img, label = iter(Train_loader)._next_data()
print(img.shape)        # torch.Size([128, 1, 28, 28])
print(label.shape)      # torch.Size([128])

# Check the Data2
# plt.figure(figsize=(10,20))
# for i in range(128):
#     plt.subplot(8,16,i+1)
#     plt.imshow(img[i,:,:,:].numpy().reshape(28,28))
#     plt.title(label[i].item())
#     plt.axis("off")
# plt.tight_layout()
# plt.show()

# Designed the NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),1)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features =1
        for s in size:
            num_features *= s
        return num_features


model = Net()

# check the parameter
params = list(model.parameters())
print(len(params))              # 10 
print(params[0].size())         # torch.Size([6, 1, 3, 3])

# 임이의 값을 만들어 forward값 확인
inputs = torch.randn(1,1,28,28)
out = model(inputs)
print(inputs.size())            # torch.Size([1, 1, 28, 28])
print(out)                      # tensor([[ 0.0021, -0.0452, -0.0633, -0.0273,  0.0708,  0.0358,  0.0917, -0.0431, 0.1449, -0.0626]], grad_fn=<AddmmBackward0>)

# 손실함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# 모델 훈련
    # optimizer.zero_grad() : 가중치의 그래디언트 초기화
    # loss.backward() : loss값 계산
    # optimizer.step() : 가중치 갱신

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(Train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%100 == 99:
            print(f"Epoch : {epoch+1}\tIter : {i}\tLoss : {running_loss}")
            running_loss=0

# 모델의 저장 및 로드
path = "./data/Model/mnist.pth"
# torch.save(model.state_dict(), path)

model2 = Net()
model2.load_state_dict(torch.load(path))
print(model2.parameters)

# 모델 테스트
dataiter = iter(Test_loader)
images, labels = dataiter._next_data()

for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i,:,:,:].view(28,28).numpy())
    plt.title(labels[i].item())
plt.show()

ouptuts = model(images)
_, pred = torch.max(ouptuts, 1)

print(pred)
print(labels)