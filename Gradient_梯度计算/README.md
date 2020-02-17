![img](https://tva1.sinaimg.cn/large/0082zybpgy1gbzw6xfd26j30rs0ij42b.jpg)

*image from unsplash.com by [@johnwestrock](https://unsplash.com/@johnwestrock)*

在深度学习中，网络参数的优化是通过 *后向传播* 实现的，而优化参数的最基本方法就是 *梯度下降* 法。使用该方法首先就要求参数对损失函数的的梯度。*梯度* 可以简单理解为参数对于损失函数的导数(实际上是偏导数)。

Tensorflow 和 PyTorch 之所以强大，是因为其自动求导的功能和自动优化能力，本文就来对比介绍 Tensorflow 和 PyTorch的自动求导和自动优化功能。



### 1. 多项式求导

为了方便理解我们这里定义一个非常简单的多项式函数：
$$
y = x^2 + x^3 + 5
$$
x 对 y 的导数为
$$
dy/dx = 2x + 3x^2
$$
当 x = 2 时 dy/dx = 2 * 2 + 3 *4 = 16。

下面我们用 Tensorflow 和 PyToch 验证一下，需要注意以下几点：

> 1. Tensorflow 中只有 Variables 才能被自动求导，如果是 constant 的张量需要 被`tape.watch()`。
> 2. 在 PyTorch 中张量需要开启 `requires_grad = True` 其计算才会被记录。
> 3. Tensorflow 中使用 `with tf.GradientTape() as tape:` 包裹并记录计算过程以便求导，PyTorch 会自动记录变量的计算过程。
> 4. Tensorflow 中 `tape.gradient(y, [x])` 返回的是一个list，与之向对应 `torch.autograd.grad(y, [x])` 返回的是元组(tuple).
> 5. PyTorch 还可以使用 y.backward() 对所有参数自动求导，然后用`.grad` 获取相应参数的导数

```python
def func(x):
    return x**2 + x**3 + 5
# df(x)/dx = 2x + 3x**2

# ------------------------Tensorflow -----------------------------
x = tf.Variable([2.])
with tf.GradientTape() as tape:
    y = func(x)
    
grad = tape.gradient(y, [x])
print(grad[0])
# ------------------------PyTorch ---------------------------------
x = torch.tensor([2.], requires_grad = True)
y = func(x)
#----method 1------
grad = torch.autograd.grad(y, [x])
print(grad[0])
#----method 2------
y.backward()
print(x.grad)
```

### 2. MSE 损失函数求导

MSE 既 Mean Square Root (均方差) 损失函数在机器学习中用得非常广泛的一个算是函数。其物理意义就是计算预测值与真实值之间的*“距离”*并取其平均数。优化模型的过程就是*”缩短“* 这个距离的过程。

这里我们假设了一个场景：

> 数据bach size 为3，输入特征维度为4，size：[3, 4], 输出 y 类别数量(depth)为2，size: [3]
> 线性变换 y = x*w + b, 待优化参数size  w:[4,2] b:[2]

我们分别用 Tensorflow 和 PyTorch 计算MSE损失函数的梯度，代码如下：

> 注意以下几点：
>
> 1. 线性变换 x@w + b 后得到的是 logits，既每个类别的 *”得分“*，需通过 softmax 函数转化属于每个类别的*"概率"* probs
> 2. 计算 probs 和 真实y 的MSE 需要将真实的 y ”one-hot“ 编码，既正确类别的概率为1，其他类别为0。 Tensorflow 自带 one-hot 编码，PyTorch 需要手动实现。
> 3. Softmax 和 MSE的实现方式在 Tensorflow 和 PyTorch 中实现的方式有多种，有方程的方式，有对象的方式，由于 Tensorflow 2.0 与 Keras 的融合，在 Keras 中也有相应的方式。
> 4. tf.losses.MSE 返回是一个一维的张量，需要用 `reduce_mean` 计算出一个标量(Scalar)。 

```python
# Example: [3,4] linear conversion ->[3,2]
#  y = x@w + b  x:[3,4] w:[4,2] b:[2], y:[3]
#  y one-hot depth = 2

# ------------------------Tensorflow -----------------------------
x = tf.random.uniform([3,4])
w = tf.random.uniform([4,2])
b = tf.zeros([2])
y = tf.constant([0, 1, 1])

with tf.GradientTape() as tape:
    # if the tensors are not variables
    tape.watch([w,b])
    
    logits = x @ w + b
    probs = tf.nn.softmax(logits)
    
    y_true = tf.one_hot(y, depth=2)
    
    losses = tf.losses.MSE(y_true,probs)
    loss = tf.reduce_mean(losses)
    
grads = tape.gradient(loss, [w,b])

grads_w = grads[0]
grads_b = grads[1]

print(loss)
print(grads[0])
print(grads[1])


# ------------------------PyTorch ---------------------------------
def one_hot(label, depth):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

x = torch.rand(3,4)
w = torch.rand([4,2], requires_grad=True)
b = torch.zeros([2], requires_grad=True)
y = torch.LongTensor([0, 1, 1])

# if "requires_grad=Flase"
# w.requires_grad_()
# b.requires_grad_()

logits = x @ w +b
probs = F.softmax(logits, dim = 1)

y_true = one_hot(y, depth=2)
loss = F.mse_loss(y_true, probs)

grads = torch.autograd.grad(loss, [w, b])

grads_w = grads[0]
grads_b = grads[1]


print(loss)
print(grads_w)
print(grads_b)

# Alternative way:

# loss.backward()
# print(w.grad)
# print(b.grad)
```

### 3. 链式法则

给一个不严谨的定义，链式法则是指在多层嵌套的函数中，内层参数对于外层函数的偏微分可以像链条一样，从外向内一步一步地求出。这使得对于无论有多深的神经网络，其每一层的参数对于损失函数的偏微分都可以通过链式法则求得。

这里我们简单举一个两次线性变换的例子

> y1 = x1 * w1 + b1
> y2 = y1 * w2 + b2

我们用 Tensorflow 和 PyTorch 验证一下 dw2/dy1 是否等于 dw2/dy2 * dy2/y1。 答案当然是肯定的。

```python
# ------------------------Tensorflow -----------------------------
x1 = tf.random.uniform([1])
w1 = tf.random.uniform([1])
b1 = tf.random.uniform([1])

w2 = tf.random.uniform([1])
b2 = tf.random.uniform([1])

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1,b1,w2,b2])
    
    y1 = x1*w1 + b1
    y2 = y1*w2 + b2
    
[dy1_dw1] = tape.gradient(y1, [w1])
[dy2_dy1] = tape.gradient(y2, [y1])
     
[dy2_dw1] = tape.gradient(y2, [w1])

print(dy2_dw1 == dy2_dy1 * dy1_dw1)

# ------------------------PyTorch ---------------------------------
x1 = torch.rand(1)
w1 = torch.rand(1, requires_grad=True)
b1 = torch.rand(1, requires_grad=True)

w2 = torch.rand(1, requires_grad=True)
b2 = torch.rand(1, requires_grad=True)

y1 = x1*w1 + b1
y2 = y1*w2 + b2

(dy1_dw1,) = torch.autograd.grad(y1, w1, retain_graph=True)

(dy2_dy1,) = torch.autograd.grad(y2, y1, retain_graph=True)


(dy2_dw1,) = torch.autograd.grad(y2, w1)

print(dy2_dy1 * dy1_dw1 == dy2_dw1)
```

### 4. 参数优化

得到参数的梯度之后，我们可以使用 *梯度下降* 的方法对参数进行优化。这里我们使用了 Himmelblau 函数。这个函数一共有 4 个零点，

> [3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.583328, -1.848126]

公式和 3d 图如下：
$$
f(x, y)=\left(x^{2}+y-11\right)^{2}+\left(x+y^{2}-7\right)^{2}
$$
![img](https://tva1.sinaimg.cn/large/0082zybpgy1gbzv46bdugj308c06975f.jpg)

*image from [wikipedia](https://en.wikipedia.org/wiki/Himmelblau%27s_function)*

#### 4.1 手动梯度下降

公式的方程代码里就略过了，我们在 Tensorflow 和 PyTorch 中分别用梯度下降的方法寻找零点。

> 注意以下几点：
>
> 1. 这里优化的参数是 x，不能直接使用 x -= lr * grad 来更新参数 ,  Tensorflow 使用`x.assign_sub(lr * grads[0])` PyTorch 使用 `x.data.sub_(lr * grads[0])`
> 2. 这里 x 的初始位置为 [0, 0], 优化后找到 [3,2] 这个点，改变初始位置会改变找到的地方。
> 3. 由于 learning rate (学习速率) 被固定在了 0.001, 所以手动梯度下降永远也到不了真正的 *零点*， 需要动态改变学习速率或者采用其他优化方法。

```python
# ------------------------Tensorflow -----------------------------
x = tf.Variable([0.,0.])
lr = 0.001

for step in range(30000): 
    with tf.GradientTape() as tape:
        pred = himmelblau(x)
        
    grads = tape.gradient(pred, [x])
    x.assign_sub(lr * grads[0])

    if(step % 2000 == 0):
        print('step {}: x = {}, pred = {}'
             .format(step, x.numpy(), pred.numpy()))

# ------------------------PyTorch ---------------------------------
x = torch.tensor([0.,0.], requires_grad=True)
lr = 0.001

for step in range(30000):
    
    pred = himmelblau(x)
    
    grads = torch.autograd.grad(pred, [x])
    x.data.sub_(lr * grads[0])
    
    if(step % 2000 == 0):
        print('step {}: x = {}, pred = {}'
             .format(step, x.tolist(), pred.item()))

```

#### 4.2 自动优化

在实际的深度学习训练中，我们完全没有必要自己手动写梯度下降的代码，Tensorflow 和 PyTorch 自带了包括梯度下降的各种优化器。

> 1. 在 Tensorflow 中，我们首先定义一个优化器 `optimizer`, 在训练过程中使用 `optimizer.apply_gradients(zip(grads, [x]))` 既可完成训练。
> 2. 在 PyTorch 中，也需要首先定义一个  `optimizer` 并指定优化参数， 在训练中，zero_grad() backward(), step() 三步即可完成训练。
> 3. SGD 既随机梯度下降，将其变换成 Adam 就会解决SGD找不到真正 零点的问题，敢兴趣的读者不妨试试。

```python
# ------------------------Tensorflow -----------------------------
x = tf.Variable([0.,0.])

lr = 0.001
optimizer = tf.optimizers.SGD(lr)

for step in range(30000):
    
    with tf.GradientTape() as tape:
        pred = himmelblau(x)
    
    grads = tape.gradient(pred, [x])
    
    optimizer.apply_gradients(grads_and_vars = zip(grads, [x]))

    if(step % 2000 == 0):
        print('step {}: x = {}, pred = {}'
             .format(step, x.numpy(), pred.numpy()))

# ------------------------PyTorch ---------------------------------
x = torch.tensor([0.,0.], requires_grad=True)

lr = 0.001
optimizer = torch.optim.SGD([x],lr=lr)

for step in range(30000):
    
    pred = himmelblau(x)
    
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    
    if(step % 2000 == 0):
        print('step {}: x = {}, pred = {}'
             .format(step, x.tolist(), pred.item()))

```

