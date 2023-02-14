import torch
x = torch.randint(1,5, size=(2,2))
y = torch.randint(1,5, size=(2,2))
print(x)
print(y)

# indexing tensor[row, col]
print(x[:,1])           # tensor([4, 4])

# View : resize, reshape the Tensor
x = torch.randn(4,5)     
y = x.view(20)
z = y.view(5,-1)
print(x.size())         # torch.Size([4, 5])
print(y.size())         # torch.Size([20])
print(z.size())         # torch.Size([5, 4])

# item : Extract the One value
x = torch.randn(1)      
print(x)                # tensor([1.2198])
print(x.item())         # 1.2197909355163574
print(x.dtype)          # torch.float32

# squeeze : minimize the Dimention
x = torch.randn(1,4,3)
s = x.squeeze()
print(x.shape)          # torch.Size([1, 4, 3])
print(s.shape)          # torch.Size([4, 3])

# unsqueeze : maximize the Dimention
x = torch.randn(1,4,3)
s = x.unsqueeze(dim=0)  # custom dimention
print(x.shape)          # torch.Size([1, 4, 3])
print(s.shape)          # torch.Size([1, 1, 4, 3])

# stack : concatenated the tensor
x = torch.randint(1,10,size=(1,5))
y = torch.randint(1,10,size=(1,5))
z = torch.randint(1,10,size=(1,5))
print(torch.stack([x,y,z]).shape)   # torch.Size([3, 1, 5])

# cat : same stack, but custom the dimention
x = torch.randint(1,10,size=(2,2,2))
y = torch.randint(1,10,size=(2,2,2))
c = torch.cat((x,y), dim=0)
print(c)
print(c.shape)          # torch.Size([4, 2, 2])

# chunk : separate the tensor (cutoms the dimention)
x = torch.randint(0,10,size=(3,6))
a,b,c = torch.chunk(x, 3, dim=1)
print(a.shape)          # torch.Size([3, 2])
print(b.shape)          # torch.Size([3, 2])
print(c.shape)          # torch.Size([3, 2])

# split : separate the tensor (cutoms the size)
x = torch.randint(0,10,size=(3,6))
a,b = torch.split(x, 3, dim=1)
print(a.shape)          # torch.Size([3, 3])
print(b.shape)          # torch.Size([3, 3])