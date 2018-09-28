## 说明

使用TensorFlow去训练自己生成的验证码

## ENVS

TensorFlow版本：1.2.1 or other

python2 / python3

## Code

* cteate_image.py：生成由大小写字母与数字组成的四位验证码

  ![验证码](./Figure/0GXh.png)

* gen_captcha.py：生成验证码，主要用于训练时调用自动生成训练集

* tensorflow_cnn_train.py：训练模型并保存，也包含测试代码

  代码中设定当精度达到0.95时终止训练

* tensorflow_cnn_test_model.py：使用保存的模型进行测试

## OTHERS


