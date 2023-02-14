# 선형회귀 모델 생성
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")

# 데이터 시각화
x = torch.randn(100,1) * 10
y = x + 3 * torch.randn(100, 1)
plt.plot(x.numpy(), y.numpy(), "o")
plt.grid()
plt.title("LR")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 신경망 구성
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        pred = self.linear(x)
        return pred

# 모델 생성 및 파라미터 확인
torch.manual_seed(0)
model = LinearRegression()
print(model)

w, b = model.parameters()
def get_params():
    return w[0][0].item(), b[0].item()

def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array([-30,30])
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(x, y)
    plt.show()
plot_fit("LR")

# set the Loss, optim
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr =1e-3)

epochs = 300
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses.append(loss)
    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        print(f"Epochs : {epoch:3d}\tloss : {loss:.4f}")

w, b = model.parameters()
plot_fit("After Train")

