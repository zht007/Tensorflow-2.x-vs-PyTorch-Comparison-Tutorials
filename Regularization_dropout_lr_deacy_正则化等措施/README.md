![UTOOLS1583965231070.png](https://img03.sogoucdn.com/app/a/100520146/d1cdcc2336560bf592e9ce383389a02f)

*Image from unsplash.com by [@tadekl](https://unsplash.com/@tadekl)*

在机器学习的过程中，**过拟合((overfiting)**是一个非常常见的问题。解决过拟合有非常多的方法，比如增加样本数量，减少模型复杂度，Early Stopping 等。在无法增加样本个数，模型不变的情况下，正则化和dropout 是处理overfiting 最常见的方法。

### 1. 正则化

模型之所会出现过拟合的情况，是因为模型死记硬背了太多训练集中数据的特征而失去了范化能力。其特征就是在训练集有很低的**损失函数**，但是对于验证数据集的分类或预测效果却比较差。

造成过拟合其中的一个原因是某些**权重**在训练的过程中被放大，而正则化就相当于给权重加了一个惩罚因子，防止其变得过大。

具体操作就是将**权重**放在**损失函数中**进行**后向传播**。由于训练的目的是减小损失函数，所以在减小损失函数的过程中，权重的大小也被减少了。L1和L2正则化就是将权重的1范数(绝对值)和2范数(平方)添加到损失函数中一起训练，在降低损失函数的过程中同样减低了权重的大小。所以正则化的过程又叫做 Weight Decay。

Tensorflow 和 PyTorch 都提供了方便的正则化的方法。

> 在 Tensorflow 中，我们调用 `keras.layers.Dense` 对象时设置`kernel_regularizer`这个参数即可。
> 在 PyTorch 中，需要在 optimizer 中设置 `weight_decay` 即可，注意这里是 L2 正则化。

```python
#---------------------------Tensorflow----------------------------

class FC_model(keras.Model):
    def __init__(self):
        super().__init__()
    
        # Regulariztion applied here
        self.model = keras.Sequential(
            [layers.Dense(200, kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.ReLU(),
            layers.Dense(100,kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.ReLU(),
            layers.Dense(10)]
            )
    
    def call(self,x):
        x = self.model(x)
        
        return x
    
#---------------------------PyTorch----------------------------------------
# L2 regularization == weight_decay in the optimizers
optimizer = torch.optim.Adam(network.parameters(),
                            lr=learning_rate, weight_decay=0.01)
```

### 2. Dropout

Dropout 就是在训练过程中，随机关闭一些神经元，从而提高模型范化能力。在Tensorflow 和 PyTorch 中均是在构建 model 的过程中在神经网络层之间插入 Dropout 层，具体来讲：

> 在 Tensorflow 中我们在构建 model 的过程中调用 `keras.layers.Dropout(dorp_rate)`
> 在 PyTorch 中 构建 model 过程中调用 `torch.nn.Dropout(drop_rate)`

```python
#---------------------------Tensorflow----------------------------
class FC_model(keras.Model):
    def __init__(self):
        super().__init__()
    
        self.model = keras.Sequential(
            [layers.Dense(200),
            layers.ReLU(),
            layers.Dropout(0.4),
            layers.Dense(100),
            layers.ReLU(),
            layers.Dropout(0.4),             
            layers.Dense(10)]
            )
    
    def call(self,x):
        x = self.model(x)
        
        return x
#---------------------------PyTorch----------------------------------------
class FC_NN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(100,10)
            )
    
    def forward(self, x):
        x = self.model(x)
        
        return x
```

### 3. 学习率下降

这里再补充一个训练神经网络的小技巧，就是 learning rate decay。 我们知道在用梯度下降训练神经网络的过程中，learning rate 越小，梯度下降得越慢。但是如果增加 learning rate 又容易造成损失函数在最小值附近徘徊不容易到达最低点的情况。

为了解决这个问题，我们可以使用学习速率下降的方法，在初期使用大的学习速率，在后期让其逐步减少从而让损失函数能够稳步探底。

在应用学习速率下降的过程中，Tensorflow 和 PyTorch的处理思路不同。在 Tensorflow 的训练过程中，我们可以用`optimizer.learning_rate` 获取当前的学习速率，或者通过赋值改变这个学习速率。换句话说就是可以在训练的过过程中干预和控制学习速率的大小。

这里我们在训练过程中提取 learning rate 并乘以一个小于 1 的下降率`lr_decay` ，知道 learning rate 小于我们设定的最小 learning rate (`lr_min`)。

```python
# set initial learning rate and minimum learning rate
lr_init = 0.2
lr_min = 1e-6
lr_decay = 0.995

optimizer = tf.optimizers.SGD(learning_rate=lr_init)

global_step = 0

for epoch in range(epochs):
    
    for step, (x, y) in enumerate(ds_train):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:            
            logits = model(x)
            
            losses = tf.losses.sparse_categorical_crossentropy(y,logits,from_logits=True)
            loss = tf.reduce_mean(losses)
            
        grads = tape.gradient(loss, model.variables)
        
        # Decay learning rate here 
        if optimizer.learning_rate > lr_min:
          optimizer.learning_rate = optimizer.learning_rate * lr_decay
        
        optimizer.apply_gradients(zip(grads, model.variables))
        
        ...
# Print current learning rate         
        
    print('accuracy: ', accuracy, 'learing rate: ', optimizer.learning_rate.numpy())
```

PyToch 采用的是另一种策略，即设置 `scheduler` 当损失函数在若干步之后不再下降时，即减小学习速率。

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200)

...
		optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate here
        scheduler.step(loss)
...

# Print current learning rate 
		print("learning rate: ", optimizer.param_groups[0]['lr'])
```



