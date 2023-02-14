## nn & nn.functional
    # 패키지가 같은 기능이지만 방식이 조금 다름
    # torch.nn은 attribute를 활용해 state를 저장하고 활용
    # torch.nn.functional로 구현한 함수의 경우 인스턴스화 시킬 필요없이 사용이 가능

# nn 패키지
    # weights, bias들이 자동으로 생성되는 레이어를 사용할 때 
        # weight값들을 직접 선언 안함
        # https://pytorch.org/docs/stable/nn.html

import torch
import torch.nn as nn

# ex : nn Conv Layer 
m = nn.Conv2d(16, 33, (3), stride=2)
m = nn.Conv2d(16, 33, (3,5), stride=(2,1), padding=(4,2))
m = nn.Conv2d(16, 33, (3,5), stride=(2,1), padding=(4,2), dilation=(3,1))

input = torch.randn(20, 16, 50, 100)
output = m(input)
print(input.shape)      # torch.Size([20, 16, 50, 100])
print(output.shape)     # torch.Size([20, 33, 26, 100])

## ex : nn.Funtional Conv Layer
import torch
import torch.nn.functional as F

filters = torch.randn(8, 4, 3, 3)
inputs = torch.randn(1, 4, 5, 5)
cov = F.conv2d(inputs, filters, padding=1)
print(inputs.shape)     # torch.Size([1, 4, 5, 5])
print(cov.shape)        # torch.Size([1, 8, 5, 5])