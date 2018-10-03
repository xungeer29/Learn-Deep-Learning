---
title: 迁移学习--Inception模型的花卉种类识别
tags: Tensorflow, 迁移学习, Inception, 花卉种类识别,
grammar_cjkRuby: true
---
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
