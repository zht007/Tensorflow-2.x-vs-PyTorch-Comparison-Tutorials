卷积神经网络在之前的文章中介绍过，这里不再赘述，主要对比一下如何使用 Tensorflow 和 PyTorch 搭建卷积神经网络(CNN)。



### 1. CNN回顾

以人脸识别问题为例，对于全连接的神经网络，每一个像素与每一个神经元相连，每一个连接都对应一个权重参数 w。我们训练这个神经网络，就相当于我们在寻找每个像素点之间的关系，随着神经网络层数的增加，参数量会爆炸式增长。

对于图片的识别最关键的是寻找局部特征，而不是寻找每个像素之间的关联，需要找到眼睛，鼻子，眼睛和鼻子的关系等等。这个时候**卷积神经网络(CNN)**就派上用场了，简单而言，CNN通过使用**滤波器(Filter)**将局部像素的"轮廓"过滤出来。同时，由于使用同一个 Filter 扫描整张图片，并且一个 Filter 的 w 又是固定的，所以相当于**共享了这些权重**。

**局部相关性**和**权重共享**这两个特性使得 CNN 在处理图片的时候非常高效。

一个 filter 有四个维度[input_channel, output_channel, (kernel__h, kernel_w)]

> 1. input_channel: 输入图片的通道数，对于一张彩色图片，通常有 3 个 channel。
> 2. output_channel: 输出图片的通道数。
> 3. kernel_ size:  filter 的二维尺寸

除此之外，CNN 计算过程中还涉及到 **填充(Padding)** ， **步长(Stride)** 和 **池化(Pooling)** 等概念，在这里就不展开讨论了。

### 2. 搭建CNN

与搭建全链接的神经网络类似，可以直接调用 Tensorflow 中的`keras.layers.Conv2D()` 对象 ，或 PyTorch 中的`nn.Conv2d()` 对象。注意 Tensorflow 和 PyTorch 存储图片的维度顺序有所不同， Tensorflow 是**[b, h, w, c] ,** PyTorch 为 **[b, c, h, w],**  其中 b, c, h, w 分别代表 batch_size, channel , height 和 width 

我们重点关注二者的区别。

> 1. `keras.layers.Conv2D()`  无需设置 input_channel 大小，然而`nn.Conv2d()` input 和 output channel需要手动设置， input_channel 需等于上一层的 output_channel
> 2. `keras.layers.Conv2D()`  中的 filters 参数相当于 `nn.Conv2d()`的 output_channel 参数。
> 3. `keras.layers.Conv2D()` 中 kernel_size 和 strides 参数接收的是元祖，`nn.Conv2d()` 中的 kernel_size 和 stride 参数接收的是 int。
> 4. `keras.layers.Conv2D()` 的 padding 参数只能选择 ‘valid’ 或 ‘same’ , 即没有padding, 或保证输出的图片尺寸不变(stride = 1的情况下)。 `nn.Conv2d()` 则需要手动设置 padding 的多少。

部分代码如下

```python

# ------------------------Tensorflow -----------------------------
class CNN_model(keras.Model):
    def __init__(self):
        super().__init__()
    
        self.model = keras.Sequential(
            [layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1),padding="same"),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.ReLU(),
            layers.Conv2D(6,(3,3),(2,2),"same"),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(10)]
            )
    
    def call(self,x):
        x = self.model(x)
        
        return x

# ------------------------PyTorch ---------------------------------
class CNN_NN(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3,6,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(6*7*7,10)
            )
    
    def forward(self, x):
        x = self.model(x)
        
        return x
```

