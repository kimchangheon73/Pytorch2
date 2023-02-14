# Autograd
    # tensor의 모든 연산에 대해 자동 미분 제공
    # 코드 작성 방법에 따라 역전파가 정의된다
    # backprop를 위한 미분값 자동 계산

# Tensor
    # data : tensor형태
    # grad : layer에 대한 미분값
    # grad_fn : 미분값을 계산한 함수에 대한 정보 저장
    # requires_grad : 텐서 내 모든 연산기록의 추척 여부
    # backward : gradient 계산
    # detach : 기록 추척 중단
    # with torch.no_grad() : weight를 고정한 채로 모델을 평가할 때 사용

# require_grad 
import torch
x = torch.ones(1,2, requires_grad=True)
y = x + 5
z = y * y* 2
out = z.mean()
print(x)            # tensor([[1., 1.]], requires_grad=True)
print(y)            # tensor([[6., 6.]], grad_fn=<AddBackward0>)
print(y.grad_fn)    # <AddBackward0 object at 0x000001A6CBB5F6D0>
print(z)            # tensor([[72., 72.]], grad_fn=<MulBackward0>)
print(out)          # tensor(72., grad_fn=<MeanBackward0>)

# require_grad
x = torch.randn(3,3)
a = ((x*3) / (x-1))
print(a.requires_grad)   # False

a.requires_grad_(True)
print(a.requires_grad)   # True

b = (a*a).sum()
print(b.grad_fn)         # <SumBackward0 object at 0x000002073A7F0220>


# Gradient : backward를 통한 역전파 계산 가능
x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() <= 1000:
    y = y*2
print(y)            

v = torch.tensor([0.1, 1, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# with torch.no_grad()
print(x.requires_grad)          # True
print((x**2).requires_grad)     # True

with torch.no_grad():
    print((x**2).requires_grad) # False


# detach
print(x.requires_grad)          # True
y = x.detach()
print(y.requires_grad)          # False
print(x.eq(y).all())            # tensor(True)

# Recall Backwoard
    # Genearl Calc : a --> b --> c --> output
    # Q : dout/da = ?

    # Back Clac :   a <-- b <-- c <-- output
    # A : dout/da = a.grad

a = torch.ones(2,2, requires_grad=True)
b = a+3
c = b**b
output = torch.sqrt(c)
print(a)                # tensor([[1., 1.],[1., 1.]], requires_grad=True)
print(b)                # tensor([[4., 4.],[4., 4.]], grad_fn=<AddBackward0>)
print(c)                # tensor([[256., 256.],[256., 256.]], grad_fn=<PowBackward1>)
print(output)           # tensor([[16., 16.], [16., 16.]], grad_fn=<SqrtBackward0>)
print(output.data)      # tensor([[16., 16.], [16., 16.]])
print(a.grad_fn)        # None(직접 계산한 부분이 없기 때문에)


# Calc the backward()
x = torch.ones(3, requires_grad=True)
y = (x**2)
z = y**2 + x            # 
out = z.sum()
print(out)              # tensor(6., grad_fn=<SumBackward0>)

grad = torch.tensor([0.1, 1, 100])
z.backward(grad)

# x.data : tensor([1., 1., 1.])   x.grad : tensor([  0.5000,   5.0000, 500.0000]) x.grad_fn :None
print(f"x.data : {x.data}\tx.grad : {x.grad}\tx.grad_fn :{x.grad_fn}")

# y.data : tensor([1., 1., 1.])   y.grad : None   y.grad_fn :<PowBackward0 object at 0x0000025BCBC98AF0>
print(f"y.data : {y.data}\ty.grad : {y.grad}\ty.grad_fn :{y.grad_fn}")

#z.data : tensor([2., 2., 2.])   z.grad : None   z.grad_fn :<AddBackward0 object at 0x000001FE7FB38AC0>
print(f"z.data : {z.data}\tz.grad : {z.grad}\tz.grad_fn :{z.grad_fn}")