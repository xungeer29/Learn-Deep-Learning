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
## ENVS
python == 2.7

tensorflow == 1.2.1

## OTHERS
数据集下载地址：[flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

模型文件下载地址：[inception_dec_2015.zip](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip)
