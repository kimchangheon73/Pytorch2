# Generate the Neural Network Construction
    # using a "torch.nn" package
    # overiding the "nn.Module" --> include the "forward" method (return the output)

# Define the Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = nn.Flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc1(x), dim=1)
        return x
    
net = Net() 
print(net)              # Net(
                        #   (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
                        #   (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
                        #   (fc1): Linear(in_features=576, out_features=120, bias=True)
                        #   (fc2): Linear(in_features=120, out_features=84, bias=True)
                        #   (fc3): Linear(in_features=84, out_features=10, bias=True)
                        # )
