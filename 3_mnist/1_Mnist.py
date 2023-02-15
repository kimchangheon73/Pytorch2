import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")


# 전처리 설정
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

# 데이터 로드 및 확인
trainset = torchvision.datasets.MNIST(root = "./data/", train = True,   download = True,    transform = transform)
testset = torchvision.datasets.MNIST(root = "./data/",  train = False,  download = True,    transform = transform)

# 데이터 적제
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

imgs ,labels = iter(train_loader)._next_data()
print(imgs.shape)
print(labels.shape)

# 데이터 확인
# plt.figure(figsize=(10,15))
# for i in range(32):
#     plt.subplot(4,8,i+1)
#     plt.imshow(imgs[i,:,:,:].resize(28,28))
#     plt.title(labels[i].item())
#     plt.axis("off")
# plt.show()

# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.flat_feature(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def flat_feature(self, x):
        feature_num = 1
        size = x.size()[1:] 
        for s in size:
            feature_num *= s
        return feature_num
        

model = Net()
print(model)

# 파라미터 확인
params = list(model.parameters())
print(len(params))          # 10
print(params[0].size())     # torch.Size([6, 1, 3, 3])

# 임의의 값을 넣어서 forward 확인
inputs = torch.randn(1,1,28,28)
out = model(inputs)
print(out)                  # tensor([[ 0.0694,  0.0436,  0.0448, -0.1248, -0.0797, -0.0460, -0.0782,  0.0504, -0.1252,  0.0665]], grad_fn=<AddmmBackward0>)

# 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# 모델 훈련 
for epoch in range(2):
    loss_sum = 0
    for idx, data in enumerate(train_loader):
        img, lab = data
        
        optimizer.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, lab)
        loss.backward()
        optimizer.step()
        
        loss_sum+=loss
        
        if idx % 100 == 99:
            print(f"Epoch : {epoch+1}\tIter : {idx+1}\tLoss : {loss_sum}")
            loss_sum=0
        

# 모델 파라미터 저장
path = "./data/model/mnist.pth"
torch.save(model, path)


# 모델 파라미터 불러오기
model2 = torch.load("./data/model/mnist.pth")

# 모델 테스트
imgs, labs = iter(test_loader)._next_data()
print(imgs.shape)       # torch.Size([128, 1, 28, 28])

outputs = model2(imgs)
value, predict = torch.max(outputs,1)
print(labs[:4])         # tensor([7, 2, 1, 0])
print(predict[:4])      # tensor([7, 2, 1, 0])

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        img, lab = data
        outputs = model2(img)
        value, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict==lab).sum().item()
        
print(f"Model Accuray : {(correct/total) * 100}")   # Model Accuray : 96.64754746835443