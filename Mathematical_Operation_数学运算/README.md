掌握 Tensorflow 和 PyTorch 的基本数学运算操作，对后续机器学习和深度学习的学习十分重要。当然两者在这方面也十分相似。

### 1. 基本数学运算

基本数学运算即“加减乘除“，Tensorflow 和 PyTorch 均使用 "+ - * /" 这四个符号，但是注意的是，这里的运算是张量元素间 (Elementwise) 的计算，张量的 Shape 必须一致, 如果有一个维度不一致，这个维度将会被 Broadcasting 成一致的 shape

```python
# ------------------------Tensorflow -----------------------------
a = tf.random.uniform([3,4])
b = tf.random.uniform([4])

print(a+b)
print(tf.add(a,b).shape)

print(a-b)
print(tf.subtract(a,b).shape)

print(a*b)
print(tf.multiply(a,b).shape)

print(a/b)
print(tf.divide(a,b).shape)

# ------------------------PyTorch ------------------------------------
a = torch.rand(3,4)
b = torch.rand(4)

print(a+b)
print(torch.add(a,b).shape)

print(a-b)
print(torch.sub(a,b).shape)

print(a*b)
print(torch.mul(a,b).shape)

print(a/b)
print(torch.div(a,b).shape)
```

### 2. 乘方开方，指数对数运算

Tensorflow 和 PyTorch 均可使用 两个星号`**` 表示乘方和开方运算，当然也可以用 `tf.pow(), torch.pow(), tf.sqrt(), torch.sqrt()` 这样的函数。

```python
# ------------------------Tensorflow -----------------------------
a = tf.fill([3,3],9.0)
print(a**2)
print(tf.pow(a, 2))
print(a**0.5)
print(tf.sqrt(a))

# ------------------------PyTorch ------------------------------------
a = torch.full([3,3],9)
print(a**2)
print(torch.pow(a,2))
print(a**0.5)
print(torch.sqrt(a))
```

指数运算对数运算两者也十分相似，唯一的区别是Tensorflow 的对数运（基底是 e）算是 `tf.math.log()`  PyTorch 是 `torch.log()`

```python
# ------------------------Tensorflow -----------------------------
b = tf.ones([3,3])*2
c = tf.exp(b)
d = tf.math.log(c)
print(b, c, d)

# ------------------------PyTorch ------------------------------------
b = torch.ones([3,3])*2
c = torch.exp(b)
d = torch.log(c)
print(b, c, d)
```

### 4. 近似计算

近似计算有一下几种情况:

> 1. floor , ceill  最小，最大取整
> 2. round 四舍五入
> 3. trunc 和 frac 取整数和小数部分

Tensorflow 没有 trunc 和 frac 函数，不过可以借助 numpy 实现

```python
# ------------------------Tensorflow -----------------------------
a = tf.random.uniform([3,3],0,10)
print(a)
print(tf.floor(a))
print(tf.math.ceil(a))
print(tf.math.round(a))
print(np.trunc(a.numpy()))
print(np.modf(a.numpy())[0])
# ------------------------PyTorch ------------------------------------
a = torch.rand([3,3])*10
print(a)
print(a.floor())
print(a.ceil())
print(a.round())
print(a.trunc())
print(a.frac())
```

### 5. 统计运算

统计运算总结连一下大致有这么几种运算。

> 1. 求范数。
> 2. 最大最小，最大最小对应的 index
> 3. 平均，求和
> 4. 最大k个数
> 5. unique 元素

需要注意的有两点，一是操作的维度( axis )，二是与 PyToch 对应的 min max sum mean 操作，Tensorflow 是 reduce_min/max/sum/mean 的操作。

```python
# ------------------------Tensorflow -----------------------------
a = tf.fill([8], 1)
a = tf.cast(a, tf.float64)
b = tf.reshape(a, [2,4])
c = tf.reshape(a, [2,2,2])

print(b)
print(c)

print(tf.norm(b,ord=1).numpy())
print(tf.norm(b,ord=2).numpy())
print(tf.norm(b,ord=2, axis=0))

a = tf.constant(np.arange(8))
a = tf.reshape(a,[2,4])
print(a)

print(tf.reduce_min(a))
print(tf.reduce_mean(a))
print(tf.reduce_max(a))
print(tf.reduce_prod(a))
print(tf.reduce_sum(a))
print(tf.argmax(a, 0))
print(tf.argmin(a, 0))

# Example: x is the output of 4 MINST images
x = tf.random.uniform([4,10])
print(tf.argmax(x,1))
print(tf.reduce_max(x, axis=1))
print(tf.reduce_max(x, axis=1, keepdims=True))
print(tf.math.top_k(x, k=5))
# ------------------------PyTorch ------------------------------------
a = torch.full([8], 1)
b = a.reshape([2,4])
c = a.reshape([2,2,2])

print(b)
print(c)

print(b.norm(1))
print(c.norm(2))

print(b.norm(1, dim = 0))
print(b.norm(2, dim = 1))

print(c.norm(1, dim = 0))
print(c.norm(2, dim = 1))
print(c.norm(2, dim = 2).shape)

a = torch.arange(8).reshape(2,4).float()
print(a)
print(a.min(), a.max(), a.mean(), a.prod())
print(a.sum())
print(a.argmax(), a.argmin())
print(a.argmax(dim=0), a.argmax(dim=1))

# Example: x is the output of 4 MINST images
x = torch.rand(4,10)
print(x.argmax(dim=1))
print(x.max(dim =1))
print(x.max(dim=1, keepdim=True))

print(x.topk(3,dim=1))
print(x.kthvalue(7,dim=1))
```

### 6. 张量比较

Tensorflow 和 PyTorch 均可使用”== , > , < , >=, <=“ 符号对两个 shape 完全相同的张量元素间 (elementwise) 进行比较，并返回与张量shape 相同的布尔值张量。

`tf.equal(), torch.eq()` 与 "==" 功能相似，但是 `torch.equal()`返回的是一个布尔值，只有比较对象完全相等才会返回 True。

### 7. 矩阵相乘

前面的运算都是张量元素间(elementwise)的运算，最后，最重要的运算就是矩阵相乘的计算。

我们数学课上学到，两矩阵(维度为2)相乘，第一个矩阵的dim = 1 的维度必须与第二矩阵 dim = 0 的维度相等，其 shape 运算规律为 [a, b] @ [b, c] ===> [a, c]。

在 Tensorflow 和 PyTorch 中我们分别使用 `tf.matul() 和 torch.matul()` 计算。当然为了书写方便，我们也社科院使用 `@` 符号代替。

当张量维度超过两个维度，计算仅对末尾两个维度进行计算。

```python
# ------------------------Tensorflow -----------------------------
a = tf.random.uniform([3,4])
b = tf.random.uniform([4,3])

print(a@b)
print(tf.matmul(a,b))

# Example: image x [2, 784] -> [2, 256]
# w: [in, out]
# b: [out]

x = tf.random.uniform([2,784])
w = tf.random.uniform([784,256])
b = tf.zeros([256])

out = x@w + b
print(out.shape)

# ------------------------PyTorch ------------------------------------
a = torch.rand(3,4)
b = torch.rand(4,3)

print(a@b)
print(torch.matmul(a,b))

# Example: image x [2, 784] -> [2, 256]
# w: [out, in],  w.t() transpose 
# b: [out]

x = torch.rand(2, 784)
w = torch.rand(256, 784)
b = torch.zeros(256)

out = x@w.t() + b
print(out.shape)
```

