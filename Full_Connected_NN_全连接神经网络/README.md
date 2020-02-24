![brown and white stripe textile](https://tva1.sinaimg.cn/large/0082zybpgy1gc7zxskh28j30rs0ijn3i.jpg)

*image from unsplash.com by [@wolfgang_hasselmann](https://unsplash.com/@wolfgang_hasselmann)*



上一篇文章，我们用 Tensorflow 和 PyTorch 分别完成了函数**自动求导**以及**参数手动和自动优化**的任务，这篇文章我们就通过经典的 MNSIT 手写数字识别数据集，对比一下,如何使用两个框架建立训练全链接的神经网络，对手写数字进行分类．



### 1. 数据导入

像 MNIST 这样经典的数据集 Tensorflow 和 PyTorch 都能直接下载，并提供非常方便快捷的加载工具．

> 1. Tensorflow 用 `tf.keras.datasets.mnist.load_data()`加载数据，数据为 `numpy.ndarray`格式 ．
>
> 2. PyTorch 从 `torchvison.datasets.MNIST` 中加载，数据格式为 Image，无法直接使用，需要设置 `transform = transforms.ToTensor()`  转换成张量数据；这里的 `transform`　不仅可以转换数据格式， 如果传入`transform.Compose()` 可以通过 list 传入更多转换的参数，比如代码中就将数据同时进行了normalize 的处理．
> 3. Tensorflow 中可以通过`tf.data.Dataset.from_tensor_slices()` 构建数据集对象．并通过 `.map`  自定义的preprocess函数，对数据进行预处理．还可以直接使用`.shuffle()`和`.batch()`对数据进行打散和批处理．
> 4. PyTorch 中使用`torch.utils.data.DataLoader` 构建数据集对象，完成数据 创建batch 批处理，以及对数据进行打散(Shuffle)
> 5. 注意处理后数据的 shape, Tensorflow 中 image shape: [b, 28, 28], label shape: [b], PyTorch  image shape: [b, 1,28, 28], label shape: [b]
> 6. PyTorch 的 DataLoader 可以设置训练数据的 `Train = False` 避免在测试数据库中对数据进行训练，而 Tensorflow 就只能在搭建网络的时候才能声明了．

```python
# ------------------------Tensorflow -----------------------------
(x, y),(x_test, y_test) = keras.datasets.mnist.load_data()

ds_train = tf.data.Dataset.from_tensor_slices((x,y))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

def preprocess(x, y):
  x = (tf.cast(x, tf.float32)/255)-0.1307
  y = tf.cast(y, tf.int32)
#   y = tf.one_hot(y,depth=10)   
  return x, y

ds_train = ds_train.map(preprocess).shuffle(1000).batch(batch_size)
ds_test = ds_test.map(preprocess).shuffle(1000).batch(batch_size)

# ------------------------PyTorch --------------------------------

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
```

### 2. 手动搭网

#### 2.1 参数初始化

我们首先介绍如何手动搭建全链接的神经网络，这里的难点是参数的初始化和管理．我们的模型有三层全链接的神经网络，所以我们需要初始化３组 w 和 b．注意每一组的shape:

> 网络：[b, 786] -> [b, 200] -> [b, 100] -> [b, 10]
>
> w1: [786, 200], b1: [200],
>
> w2: [200,100], b2: [100],
>
> w3: [100,10], b3：[10]

```python
# ------------------------Tensorflow -----------------------------
w1 = tf.Variable(tf.random.uniform([28*28, 200]))
b1 = tf.Variable(tf.zeros([200]))

w2 = tf.Variable(tf.random.uniform([200, 100]))
b2 = tf.Variable(tf.zeros([100]))

w3 = tf.Variable(tf.random.uniform([100, 10]))
b3 = tf.Variable(tf.zeros([10]))
# ------------------------PyTorch --------------------------------
w1 = torch.rand(28*28, 200 , requires_grad=True)
b1 = torch.zeros(200, requires_grad=True)

w2 = torch.rand(200, 100, requires_grad=True)
b2 = torch.zeros(100, requires_grad=True)

w3 = torch.rand(100, 10, requires_grad=True)
b3 = torch.zeros(10, requires_grad=True)
```

#### 2.2 搭建网络

这里我们均采用自定义函数的方式来搭建网络，这个部分两个框架没有太大区别．我们手动定义了三层神经网络，前两层包含 relu 激活函数，最后一层没有使用激活函数．

```python
# ------------------------Tensorflow -----------------------------
# forward func
def model(x):
    x = tf.nn.relu(x@w1 + b1)
    x = tf.nn.relu(x@w2 + b2)
    x = x@w3 + b3
        
    return x
# ------------------------PyTorch --------------------------------
# forward func
def forward(x):
    x = F.relu(x@w1 + b1)
    x = F.relu(x@w2 + b2)
    x = x@w3 + b3
        
    return x
```

#### 2.3 训练网络

该部分与前文中介绍的自动求导，参数优化的部分一致，按照套路进行就行了，需注意以下几点．

> 1. 对于全链接网络首先需要对数据打平，Tensorflow 和 PyTorch 都可以用 reshape 方法实现．
> 2. 为了与 PyTorch 中`torch.nn.CrossEntropyLoss()`求**交叉熵**的方法一致，Tensorflow 中并未对label 进行 One-Hot 编码，所以使用了`tf.losses.sparse_categorical_crossentropy()` 方法计算交叉熵．

```python
# ------------------------Tensorflow -----------------------------
optimizer = tf.optimizers.Adam(learning_rate)

for epoch in range(epochs):
    
    for step, (x, y) in enumerate(ds_train):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:            
            logits = model(x)
            
            losses = tf.losses.sparse_categorical_crossentropy(y,logits,from_logits=True)
            loss = tf.reduce_mean(losses)
            
        grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])
        
        optimizer.apply_gradients(zip(grads, [w1,b1,w2,b2,w3,b3]))
# ------------------------PyTorch --------------------------------
optimizer = torch.optim.Adam([w1,b1,w2,b2,w3,b3],
                            lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    
    for step, (x, y) in enumerate(train_loader):
        x = x.reshape(-1,28*28)
        
        logits = forward(x)
        loss = criteon(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. 高级 API 搭建网络

手动搭建网络的好处是，都是采用最底层的方式，整个过程透明可控．但是坏处就是需要手动管理每一个参数，网络一旦复杂起来就容易出错．

Tensorflow 和 PyTorch 均可采用创建模型*对象(Class)*的方式创建神经网络模型．

> 1. Tensorflow 继承 `tf.keras.Model`对象，PyTorch 继承 `torch.nn.Module`对象．
> 2. Tensorflow 模型对象中，前向传播调用 `call()` 函数，PyTorch 调用 `forward()` 函数．
> 3. 在训练过程中仅需将手动搭网的函数替换成初始化后的网络模型对象即可．

```python
# ------------------------Tensorflow -----------------------------
class FC_model(keras.Model):
    def __init__(self):
        super().__init__()
    
        self.model = keras.Sequential(
            [layers.Dense(200),
            layers.ReLU(),
            layers.Dense(100),
            layers.ReLU(),
            layers.Dense(10)]
            )
    
    def call(self,x):
        x = self.model(x)
        
        return x
    
model = FC_model()
# ------------------------PyTorch --------------------------------
class FC_NN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100,10)
            )
    
    def forward(self, x):
        x = self.model(x)
        
        return x

network = FC_NN().to(device)  
```

### 4. 使用 GPU 加速训练

如果训练环境支持 GPU ，Tensorflow 和 PyTorch 均可以调用 GPU 加速计算．Tensorflow 如果使用的是 Tensorflow-gpu 版本，我们无需任何操作，直接就是调用的GPU进行计算．

对于 PyTorch ，需要创建 `device = torch.device('cuda:0')`并将网络和参数搬到这个 device 上进行计算．

```python
...
device = torch.device('cuda:0')
network = FC_NN().to(device)  

criteon = torch.nn.CrossEntropyLoss().to(device)
...

for epoch in range(epochs):
...        
        x, y = x.to(device), y.to(device)
...
```

### 5. 模型测试

模型训练好了之后需要使用验证数据集进行测试。这里我们简单的采用**正确率(accuracy)**来对模型进行验证

> 正确率 = 预测正确的样本数 / 所有样本数

代码看起来比较繁琐，不过就是以下几个步骤：

> 1. 将所有验证数据带入训练好的模型中，给出预测值。
> 2. 将预测值与实际值进行比较。
> 3. 累加预测正确的样本数和总样本数。
> 4. 用上面的公式算出正确率

实际上 tensorflow 可以调用`tf.keras.metrics` 这个在之前的文章中已经提到，这里就不赘述了。

```python
# ------------------------Tensorflow -----------------------------
if(step%100==0):
            print("epoch:{}, step:{} loss:{}".
                  format(epoch, step, loss.numpy()))         
            
#             test accuracy: 
            total_correct = 0
            total_num = 0
            
            for x_test, y_test in ds_test:
                x_test = tf.reshape(x_test, [-1, 28*28])
                y_pred = tf.argmax(model(x_test),axis=1)
                y_pred = tf.cast(y_pred, tf.int32)
                correct = tf.cast((y_pred == y_test), tf.int32)
                correct = tf.reduce_sum(correct)
                
                total_correct += int(correct)
                total_num += x_test.shape[0]
        
            
            accuracy = total_correct/total_num
            print('accuracy: ', accuracy)
# ------------------------PyTorch --------------------------------
        if(step%100 == 0):
            print("epoch:{}, step:{}, loss:{}".
                  format(epoch, step, loss.item()))
        
#             test accuracy
            total_correct = 0
            total_num = 0    

            for x_test, y_test in test_loader:
                    x_test = x_test.reshape(-1,28*28)
                    x_test, y_test = x_test.to(device), y_test.to(device)

                    y_pred = network(x_test)
                    y_pred = torch.argmax(y_pred, dim = 1)
                    correct = y_pred == y_test
                    correct = correct.sum()

                    total_correct += correct
                    total_num += x_test.shape[0]

            acc = total_correct.float()/total_num
            print("accuracy: ", acc.item())
```

