### 1. 初始化

Tensorflow 和 PyTorch 的导入，查看版本，是否有GPU支持等非常简单代码如下：

![image-20200204112424027](https://tva1.sinaimg.cn/large/006tNbRwgy1gbklazmyxuj312w086jse.jpg)

### 2. 创建张量

在数学和物理概念中，数据有，**标量 Scaler，向量 Vector, 矩阵 Matrix 和张量 Tensor** 之分。但我们这里为了简单和统一，将数据均称为张量。

Tensorflow 和 PyTorch 张量初始化可以直接分别调用：` tf.constent` 方法，调用 `torch.tensor` 方法，填入张量数值即可。

也可分别创建 `tf.Variable`, `torch.Tensor` 对象，实现张量初始化。但是这里需要注意两点:

> 1.  `tf.Variable` 创建的是` Variable `对象 ，不是` Tensor `对象，前者可以跟踪求梯度，后者不能直接求梯度。
> 2. `torch.tensor `和 `torch.Tensor` 均创建的是 `Tensor` 对象，但前者输入具体数值，后者输入 Tensor shape(size)，数值不可控，不推荐

推荐初始化方法

> 1. 是直接从 numpy.arrary 转换成 Tensor 
> 2. 使用，zeros, ones 和 eye 分别创建 0， 1 和 对角张量。
> 3. 随机初始化 `tf.random.uniform`,` tf.random.norma`l; `torch.rand`

均可以通过 `.numpy()` 转换成 `numpy.arrary`

![image-20200204114718775](https://tva1.sinaimg.cn/large/006tNbRwgy1gbklysob9vj312k0rsn1z.jpg)

### 3. 切片和索引

Tensorflow 和 PyTorch 切片和索引与 numpy.array 非常类似这里就不赘述了。注意掌握以下两个原则：

> 1. 索引格式 **start​ : end : step​** 其中 end 不包括在内。
> 2. 中间若要省略过多冒号，可用 ... 代替

### 4. 维度变换

#### 4.1 Reshape

Tensorflow 和 PyTorch 均可用 `reshape` 方法变换维度。PyTorch 还有一个 `view` 方法，与`reshape` 方法效果一样。

> 需要注意的是
>
> 1. Tensorflow 是调用 tf.reshape, PyTorch 可以直接对 Tensor 本身进行操作。 
> 2. Reshape 操作不会改变数据的存储顺序。

另外可以 Tensorfow 和 PyTorch 可以分别使用` tf.expand_dim` 和 `.unsqueeze` 方法增加维度。分别使用 `tf.squeeze` 和 `.squeeze `方法删除维度。

> 注意增加和删除的维度在该维度 (Aix) 上数值为 1，表现在数据上就是增加和删除了括号而已，并不会对数据的存储顺序造成影响。

#### 4.2 交换维度

交换维度是非常常见的操作，比如 Tensorflow 中图片的存储方式通常是 [b, w, h, c] 而 PyTorch 则为 [b, c, w, h]

> b, w, h, c 分别代表 batch, width, height, channel. （批，宽，高，通道）

Tensorflow 和 PyTorch 均可使用 `transpose` 的方法交换维度，但是实现方法有所差异，PyTorch 推荐使用 `permute` 方法，该方法与` tf.transpose` 方法一致。

>  交换维度会改变数据存储顺序，谨慎使用。

该部分参考代码如下：

![image-20200204122754642](https://tva1.sinaimg.cn/large/006tNbRwgy1gbkn4zg7j7j30p109k76n.jpg)



### 5. Broadcasting

Broadcasting 是一个自动复制张量的手段，在深度学习中，通常我们需要将标量的偏置 (Bias) Broadcast 到与权重 (Weight) 一致。

Tensorflow 和 PyTorch 均可自动完成这个操作。 当然我们也可以分别使用 `tf.broadcast_to`,`.expand` 方法手动实现。

> 注意 Broadcasting 只能将维度(轴)上数值为1的地方复制 n 遍。

![image-20200204123843502](https://tva1.sinaimg.cn/large/006tNbRwgy1gbkng8hlzlj313406u0u2.jpg)

### 6. 合并拆分

合并和拆分是非常常见的张量操作，对应的是 concat 和 split。Tensorflow 对应 `tf.concat`和`tf.split` PyTorch对应 `.cat` 和`.split`，这两个操作都不会改变张量的维度。

举一个例子，一个3维张量对应 班级，学生，各科成绩 [classes, students, scores] ，我们可以在三个维度上对这个张量进行合并和拆分。

> 如果在 dim/axis=0 上合并，既合并班级 ，此时 students, scores 维度上数值必须保持一致，既学生数量和科目数量必须保持一致。
>
> 如果在 dim/axis=1 上合并，既合并学生 ，此时 classes, scores 维度上数值必须保持一致，既班级数量和科目数量必须保持一致。
>
> 如果在 dim/axis=0 上合并，既合并成绩 ，此时 classes, student 维度上数值必须保持一致，既班级数量和学生数量必须保持一致。

拆分也是同样的道理，在传入参数的 `num_or_size_splits` 如果传入一个数字，既表示在该维度下每份的大小，如果传入的是一个 list, 既表示按照 list 中的大小进行拆分。

![image-20200205154552775](https://tva1.sinaimg.cn/large/006tNbRwgy1gblyhcssmkj30ou0cpn07.jpg)

另外还有一个概念需要区分就是 stack 和 unstack

> stack 会在相应维度前面新加一个维度，所以被 stack 的张量在每个维度上的大小必须一致。
>
> tf.unstack 会将该维度完全拆散，既[3,32,8] -> axis=0 下 ->unstack 为三个[32,8]的矩阵。
>
> torch.chunk 与split 概念一致，是指在该维度下**均分**为多少份

