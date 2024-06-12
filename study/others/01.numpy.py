import numpy as np

n1 = np.array([1, 2, 3])
print(n1)
n2 = np.array([[1, 2, 3], [2, 4, 6]])
print(n2)
print(n2.shape)
print(n2.max())

print(n2.min)

# 浅拷贝，n1 n2 地址一样
n1 = np.array([1, 2, 3])
n2 = n1
n2[0] = 2
print(n1, n2)
print(id(n1), id(n2))
# [2 2 3] [2 2 3]
# 2584770317840 2584770317840


# 深拷贝
n2 = np.array(n1, copy=True)
n2[0] = 9
print(n2)
print(id(n1), id(n2))
# [9 2 3]
# 2369907789328 2369907789520


n1 = [1, 2, 3]
n2 = np.array(n1, ndmin=3)
print(n2)
print(n2.shape)

n1 = np.zeros((1, 3))
n1 = np.full(shape=(3, 3), fill_value=10)

print(n1)
print(n2)
print(n1 + n2)
print(n1 - n2)

n1.reshape(1, 9)
print(n1)

a = np.mat([[1, 2], [3, 4]])
b = np.mat([[1, 2], [3, 4]])
c = a + b
print(c)
print(a * b)
print(np.multiply(a, b))  # 点乘

print(a.T)  # 转置

print(a.I)  # 求逆

print(np.add(a, b))  # 加
print(np.subtract(a, b))  # 减法
print(np.divide(a, b))
print(np.power(a, 3))  # 求指数

print(a.sum())  # 求和
print(a.sum(axis=0))  # lie求和
print(a.sum(axis=1))  # hang求和
print(a.mean())
print(a.mean(axis=0))
print(a.mean(axis=1))

print(a.min(axis=0))
print(a.max(axis=1))

print(np.sort(a))
# 中位数
print(np.median(a, axis=0))
# 方差
print(np.var(a))
# 标准差
print(np.std(a))
