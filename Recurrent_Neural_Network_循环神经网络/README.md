![person standing on rock formation during daytime](https://tva1.sinaimg.cn/large/00831rSTgy1gd83b5opk0j30rs0fm40b.jpg)

*image form unsplash.com by [@chrishenryphoto](https://unsplash.com/@chrishenryphoto)*

循环神经网络 (RNN) 主要处理时间序列的数据，比如语音，文字等等。循环神经网络的基本概念在之前的文章中已经介绍过了，本文主要是比较一下 Tensorflow 和 PyTorch 中如何搭建和训练循环神经网络。



### 1. RNN 回顾

与处理**空间相关性**的 CNN 相对应，RNN 主要用于处理**时间相关性**的问题。以分类影评的问题为例(即判断影评是正面还是负面 IMDB 数据集)，我们当然也可以使用全连接的神经网络，将里面每一个单词都连接起来，不过这样的话，模型既占用大量计算资源，又无法感知前后语句的相关性。

在 RNN 中，一句话的每个单词，共享相同结构的神经网络和相应的权值，引入**状态张量 （h）**，将每个单词的输入与输出连接起来，就很好地解决了全连接神经网络的问题，同时也让神经网络记录了前后语意。

> **局部相关性**和**权重共享**这两个特性使得 CNN 在处理图片的时候非常高效。
> **前后相关性**和**权重共享**这两个特性使得 RNN 在处理时间序列的问题时非常高效。

![image-2020032594619409](https://tva1.sinaimg.cn/large/00831rSTgy1gd6w98cpctj30ka09cjrw.jpg)

对初学者来说，理解 RNN 最困难的地方是，神经网络需要数据的在时间维度展开。

> 例如:
>
> * 图片 shape 为 [b, w, h, c]， 我们可以理解为在 b (batch) 维度展开，将[w, h, c] 这样的图片输入神经网络。
> * 句子 shape 为[b, seq_len, embeded_dim] (seq_len 即句子中单词的数量，embeded_dim 指单词编码的维度)  需要在 seq_len 这个维度展开，将每一个单词送入神经网络。

### 2. 数据导入

在 Tensorflow 中，我们可以直接在 `keras.datasets.imdb`中导入IMDB的数据库，并转化成Dataset 对象进行预处理，关于Tensorflow 2.0 数据导入和处理的部分参见前面[这篇文章](https://www.jianshu.com/p/b796823ad32c)。这里注意两点

> 1. 数据已经进行了编码，所以数据类型是 int 而不是 string.
> 2. 使用了 `keras.preprocessing.sequence.pad_sequences `将每个句子的长度 (seq_len) 进行了固定。

在 PyTorch 中， 我们从 `torchtext.datasets.IMDB` 中获取，注意需要将数据的 train_data 和 test_data 进行分离，并且初始化清楚文本(TEXT) 和 标签(LABEL)对象。

```python
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
```

由于这里文本的数据类型是 string 需要编码成 int 以便进行计算，这里通过 glove 进行编码，同时确定 batch size。

```python
TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

batchsz = 64
device = torch.device('cuda')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = batchsz,
    device=device
)
```

> 这里需要特别注意：
>
> 在 Tensorflow 输入数据的 shape 为 [b, seq_len] 经过 embedding 层之后为 [b, seq_len, embeding_dim]
> 在 PyTorch 中输入数据的 shape 为 [seq_len, b] 经过 embedding 层之后为[seq_len, b, embeding_dim]
> 在 Tensorflow 中，seq_len 是固定的，在 PyTorch 中虽然每个 batch 的 seq_len 不一样，但是同一个batch 内的 seq_len 是相同的

### 3. RNN 神经网络搭建

首先，在 Tensorflow 中调用 `keras.layers.Embedding` 对象，在 PyTorch 中调用 `torch.nn.Embedding`对象对输入进行编码。

>  其中相对应的参数：
> `input_dim` 与`num_embeddings`相对应，即表示字库容量。
> `output_dim` 与 `embedding_dim`相对应，即表示编码维度。

此时将数据的 shape:

> Tensorflow: 由 [b, seq_len] 变为 [b, seq_len, embedding_len]
> PyTorch: Tensorflow: 由 [seq_len, b] 变为 [seq_len, b, embedding_len]



其次， 在 Tensorflow 中调用 `keras.layers.SimpleRNN` 对象，在 PyTorch 中调用 `torch.nn.RNN`对象搭建 RNN 层。
这里需要注意的是

>  `keras.layers.SimpleRNN` 一次只能搭建一层 RNN 层，如果要叠加多层，参数 `return_sequences`需要打开以传递中间的**状态张量**。输出为最后一层的最后一个单词的output，shape 为 [b, units] (unitis 为 神经元个数与 PyTorch 中的 hidden_dim 相对应)
>
> `torch.nn.RNN` 一次可以搭建多层 RNN 层，同时输出 output 和 **状态张量** 。output 包含了每个单词的输出，其shape 为[seq_len, b, hidden_dim]; **状态张量** 的是每层最后单词的output 其shape 为[num_layers, b, hidden_dim]。(hidden_dim 为神经元个数)



最后通过全连接层输出分类结果。但是需要注意的是在处理 IMDB 的问题中，我们需要最后一层的最后一个单词的输出 (Tensorflow 是直接给出这个输出的)，所以在 PyTorch 中需要对数据进行切片。

代码如下

```python
# ---------------Tensorflow-------------
class RNN_model(keras.Model):
    def __init__(self, num_units):
        super().__init__()
    
        # embedding   [b, 80] ->[b, 80 embedding_len=100]  
        self.embedding = layers.Embedding(input_dim=total_words,output_dim=embedding_len,input_length=max_review_words)
        # [b, 80, 100] ->[b, num_units]
        self.RNN1 = layers.SimpleRNN(units=num_units,dropout=0.5, return_sequences=True)
        self.RNN2 = layers.SimpleRNN(units=num_units,dropout=0.5, return_sequences=False)
        
        # [b, num_units] ->[b,1]
        self.fc = layers.Dense(1)
    
    def call(self,x, training = None):
        outputs = self.embedding(x)
        outputs = self.RNN1(outputs, training = training)
        outputs = self.RNN2(outputs, training = training)
        outputs = self.fc(outputs)
        return outputs
# ---------------PyTorch----------------
class RNN_nn(nn.Module):

  def __init__(self, vocab_size, hidden_dim):
    super().__init__()

    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_len)
    self.rnn = nn.RNN(input_size=embedding_len, hidden_size=hidden_dim, num_layers=2, dropout=0.5)
    self.fc = nn.Linear(in_features=hidden_dim, out_features=1)

  def forward(self, x):  

    
    #[seq_len, b] -> [seq_len, b, embedding_len = 100]
    output = self.embedding(x)
    # print('embedding size: ', output.shape)
    
    #[seq_len, b, embedding_len] ->
    # out: [seq_len, b, hidden_dim]
    # h:   [n_layer=2, b, hidden_dim]
    # last seq of out == last layer of h
    output, h = self.rnn(output)


    # Last seq of the output
    out = output[-1:,:,:].squeeze()
    
    out = self.fc(out)

    return out
```

神经网络的训练和验证部分与全连接神经网络以及CNN非常类似，这里就不赘述了，感兴趣的读者请参考源代码。

