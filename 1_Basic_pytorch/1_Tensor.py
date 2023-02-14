# 파이토치의 구성요소
    # torch : 텐서 생성 도구
    # torch.autograd : 자동미분 도구
    # torch.nn : 신경망 생성 도구
    # torch.multiprocessing : 병렬처리 도구
    # torch.utils : 데이터 조작 도구
    # torch.onxx : 다른 프레임워크 간 모델 공유 도구

# 텐서 : n차원의 배열 개념과 유사, gpu연산 가속도 가능
import torch
print(torch.__version__)

# Empty Tensor (row,col)
x = torch.empty(4,2)
print(x)

# Random Tensor(row,col)
x = torch.rand(4,2)
print(x)

# Zero Tensor( dtype = long )
x = torch.zeros(4,2, dtype=torch.long)
print(x)

# custom TEnsor
x = torch.tensor([4,2,3,1,4])
print(x)

# Change One Tensor
x = x.new_ones(2,4,dtype=torch.double)
print(x)

# same shape, differ type Tensor(other tensor, dtype = ?)
x = torch.rand_like(x, dtype=torch.float)
print(x)

# Tensor Shape
print(x.size())
print(x.shape)