import torch
import numpy as np

print(torch.cuda.is_available())

t1 = torch.tensor([1, 2, 3])
print(t1)
print(t1.dtype)

# torch.set_default_tensor_type(torch.DoubleTensor) 更改默认的tensor数据类型
print(t1.dtype)

array = np.array([[5, 6], [7, 8]])
print(array)
print(array.dtype)
t2 = torch.tensor(array)
print(t2)
print(t2.dtype)

t1 = torch.DoubleTensor([2])
t2 = torch.IntTensor([3])
print(t1, t2)

t1 = torch.zeros(size=(3, 3), dtype=torch.float32)
t2 = torch.ones(size=(3, 3))
print(t1, t2)

t3 = torch.ones_like(input=t1)
print(len(t3))
print(type(t3))
print(t3)

# 生成同纬度的随机张量

t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
# t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
rand = torch.rand_like(input=t1)
print(rand)

# 自定义填充

t1 = t1.new_full((3, 4), 3.141592)
print(t1)
print(t1.dtype)
print(t1.int().dtype)
print(t1.to(dtype=torch.int32).dtype)

print(t1[0][0])
print(t1[0][:])
print(t1[0][0].item())

# 装量转数组
print(t1.numpy())
# 数组转张良
print(torch.Tensor(t1.numpy()))
print(torch.as_tensor(t1.numpy()))
print(torch.from_numpy(t1.numpy()))

# user_name = "黑马程序员"
# user_type = "SSSSSVIP"
user_name = input()
user_type = input()
print(f"您好{user_name},您是最贵的{user_type}用户，欢迎您")
print("您好user_name,您是最贵的user_type用户，欢迎您")



