# Calc of Tensor

# Set the Variable
import torch
x = torch.randint(0,5, size=(2,2))
y = torch.randint(0,5, size=(2,2))
print(x)
print(y)

# Add / sub / mul / div --> the Same method
result = torch.add(x,y)
print(result)

# return the output to Variable
result2 = torch.empty(2,2, dtype=torch.int)
torch.add(x,y, out=result2) 
print(result2)

# inplace Calc
print(x)        # tensor([[3, 3],[1, 2]])
x.add_(y)
print(x)        # tensor([[3, 6],[4, 4]])

# Dot product 
result3 = torch.mm(x,y)
print(result)   # tensor([[3, 6],[4, 4]])
