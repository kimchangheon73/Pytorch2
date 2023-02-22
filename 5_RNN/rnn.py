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

class rnn(nn.Module):
    def __init__(self, input, hidden, output):
        super(rnn, self).__init__()
        self.hidden = hidden
        self.h1 = nn.RNN(input, self.hidden)
        self.h2 = nn.RNN(self.hidden, output)

    def forward(self, input):
        x = input.view(-1,28*28)
        x, _ = self.h1(x)
        x = F.tanh(x)
        x, _ = self.h2(x)
        return x
        



# 입력 크기가 28*28, 히든 100, 10
x = torch.randn(128,784)        # torch.Size([128, 1, 28, 28])

r1 = nn.RNN(28*28, 100)             #   (0): RNN(784, 100)
r2 = nn.RNN(100, 10)                #   (1): RNN(100, 10)
r = nn.Sequential(r1,r2)

output, _status = r1(x)             # torch.Size([128, 100])
output2, _status2 = r2(output)      # torch.Size([128, 10])

# 모델 테스트
# 전처리 설정
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])

# 데이터 로드 및 확인
trainset = torchvision.datasets.MNIST(root = "./data/", train = True,   download = True,    transform = transform)
testset = torchvision.datasets.MNIST(root = "./data/",  train = False,  download = True,    transform = transform)

# 데이터 적제
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

# 데이터 확인 
imgs ,labels = iter(train_loader)._next_data()
print(imgs.shape)
print(labels.shape)

# Set Model
model = rnn(784,500,10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

print(model)

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
        