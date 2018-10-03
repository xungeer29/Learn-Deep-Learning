# 迁移学习--Inception模型的花卉种类识别

## 文件结构
```
-transfer_learning
	-flower_photos	//花卉图片
		-daisy		
		-dandelion		
		-roses		
		-sunflowers		
		-tulips		
	-model
		-tensorflow_inception_graph.pb   //模型文件
		-imagenet_comp_graph_label_strings.txt
	-bottleneck   //保存模型瓶颈层的特征结果
		-daisy		//daisy类花特征保存在txt中
		-dandelion
		-roses
		-sunflowers
		-tulips
	-flower_recognize.py
```

## Code
* 先将图片过一遍inception-v3,　得到BOTTLENECK_TENSOR_NAME的权重，然后自己在加了一个分类的全连接层，
定义了损失函数与优化方法，得到最终的分类输出；

* bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

  将图像过一遍inception-v3，得到图片的 feature map;
 
* gfile.FastGFile(image_path, 'rb').read()

　读取图像，可以直接处理成tensorflow需要的张量形式，如果使用opencv等读取，还需要转化成tensorflow内部的张量形式；


## ENVS
python == 2.7

tensorflow == 1.2.1

## OTHERS
数据集下载地址：[flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

模型文件下载地址：[inception_dec_2015.zip](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip) (需要翻墙)

知乎lqfarmer写的上关于迁移学习的[综述](https://zhuanlan.zhihu.com/p/27368456)(引用不正规，如有侵权，请联系删除！)
