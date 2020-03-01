之前讨论 Tensorflow 2.0 的文章中我们介绍了其非常强大的可视化工具：Tensorboard。虽然 PyTorch 中我们可以使用非官方的 Tensorboarx 来实现相同的功能，但是我们更推荐 Facebook 的可视化工具 Visdom。这篇文章我们就通过对比 Tensorboard 的形式介绍 Visdom。

本文中数据导入和模型建立与上一篇文章中的代码一致就不在这里赘述了。

### 1. Tensorboard 回顾

简单来说，tensorboard 就是通过监听定目录下的 log 文件然后在 Web 端将 log 文件中需要监听的变量可视化出来的过程。

所以，使用 Tensorboard 大致分为以下三步：

> 1. 创建监听目录 `logdir`
> 2. 创建 `summary_writer` 对象写入 `logdir`
> 3. 将数据写入到 `summary_writer` 中

查看 Tensorboard 需要在终端(windows cmd) 中 cd 到当前目录并输入以下命，其中 'logs' 就是我们需要监听的目录。

```
tensorboard --logdir logs
```

终端会返回一个 url，我们在浏览器中打开这个 url 即可看到 Tensorboard 了。

### 2. Visdom 使用简介

Visdom 同样也需要在浏览器中查看，与Tensorboard 不同，我们并不需要事先创建一个 log 目录，仅需要在终端 (Windows CMD) 中输入如下命令，建立visdom 环境的服务器。

```
python -m visdom.server
```

终端会返回一个 url, 我们在浏览器中打开这个 url 即可看到 Visdom 的环境了。 下面我们通过与 Tensorboard 的对比学习 Vidsom 。

> 1. Tensorboard 创建 `summary_writer=tf.summary.create_file_writer(log_dir)`对象；相应的 Visdom 中创建` vis = Visdom()`对象。
> 2. Tensorboard 使用 `with summary_writer.as_default():` 包裹 `tf.summary.scalar` 监听 scalar 的变化；同样的，Visdom() 中使用 `vis.line` 将监听对象的变化用 line 画出来。
> 3. Tensorboard 中 `tf.summary.scalar` 的参数 data 和 step 对应 `vis.line` 中的参数 Y 和 X。
> 4. 注意 vis.line 中的 Y 和 X 接受的是一个list，我们可以在这个list 中添加多个监听对象从而实现一个坐标图中监听多个变量。
> 5. Tensorboard 中使用`tf.summary.image` 显示图片其shape 为[b, w, h ,c]，Visdom 使用`vis.images` 显示图片图片shape 为[b, c, w, h]，

### 3. 代码对比

最后我们还是在代码中对比两者的异同吧 (仅展示部分代码)

```python
# -------------------------Tensorflow with Tensorboard-------------------------
import datetime, os
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

class FC_model(keras.Model):
....
globle_step = 0    
for epoch in range(epochs):
    
    for step, (x, y) in enumerate(ds_train):
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:            
            ...
            loss = tf.reduce_mean(losses)
            
            with summary_writer.as_default():            
                tf.summary.scalar("train_loss_each_step",loss.numpy(),step=globle_step)            
		...
        if(step%100==0):
#             test accuracy: 
            total_correct = 0
            total_num = 0
            
            for x_test, y_test in ds_test:
				...
                loss_test = tf.reduce_mean(loss_test)
                y_pred = tf.argmax(logits_test,axis=1)
				...
            
            accuracy = total_correct/total_num
            
            x_images = x_test[:25]
            x_images = (x_test+0.1307)*255
            x_images = tf.reshape(x_images, [-1,28,28,1])
            
            with summary_writer.as_default():
                tf.summary.scalar("accuracy", accuracy, step=globle_step)
                tf.summary.scalar("train_loss", loss.numpy(),step=globle_step)
                tf.summary.scalar("test_loss", loss_test.numpy(),step=globle_step)
                
                tf.summary.image("images",x_images,step=globle_step,max_outputs=25)
        
        
        globle_step += 1

# ------------------------------PyTorch with Visdom------------------
from visdom import Visdom

vis = Visdom()

class FC_NN(nn.Module):
...

globle_step = 0

for epoch in range(epochs):
    
    for step, (x, y) in enumerate(train_loader):
        ...
        loss = criteon(logits, y)
        
        vis.line([loss.item()],[globle_step],win='loss_each_step',
                 update='append',opts=dict(title='train_loss_each_step') )
        
		...
        
        if(step%100 == 0):

            total_correct = 0
            total_num = 0    

            for x_test, y_test in test_loader:
					...
                    loss_test = criteon(logits_test, y_test)
                    y_pred = torch.argmax(logits_test, dim = 1)
					...

            acc = total_correct.float()/total_num
            print("accuracy: ", acc.item())
            
            vis.line([acc.item()],[globle_step],
                     win='acc', update='append',
                    opts=dict(title = 'accuracy'))
            
            vis.line([[loss.item(), loss_test.item()]],[globle_step], 
                             win='losses', update='append', 
                             opts=dict(title='losses',legend=['train_loss', 'test_loss'] ))
            
            vis.images(x_test.reshape(-1,1,28,28), win='images')
            vis.text(str(y_pred.detach().cpu().numpy()), win = 'pred')
    
        globle_step += 1           

```

### 4. 总结

Tensorboard 和 Visdom 原理类似，各有优缺点。个人感觉 Visdom 更为简洁和高效，并且图表刷新速度也比 Tensorboard 快很多。

另外，在 Tensorflow 中仍然可以使用 Visdom，此时仅需要将监听的对象转换成 numpy 格式即可，这部分内容请见完整的源代码。