# ShatheNet

This subproject is focused on learning more about convolutional neuroanal networks by building my own CNN. Starting from a very simplistic CNN, I'll try to get the higher accuracy on a certain dataset.

## Datasets
[Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html): This is the data-set I am going to work with. 50k training images and 10k test iamges. 10 classes. 32 x 32 pixels although they are upsampled while training.

The other dataset I will work with is [Caltech-256 Object Category Dataset](http://authors.library.caltech.edu/7694/). A set of 256 object categories containing a total of 30607 images. 
Its size is 1GB. I use this dataset instead of Iamgenet due to its size which makes the development and test easier and faster.


## Comparisons
### Cifar-10

| Version        | Notes           | Params           | Cifar-10 Accuracy |
| ------------- |:-------------:|:-------------:| -----:|
| [1.0](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_0.png)     | First simple approach   | 17K   | 60% |
| [1.1](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_2.png)     | + convs. +Params  | 1.25M   | 76% |
| [1.2](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_1.png)     | + convs. - Params  | 930K   | 83% |
| [1.3](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_3.png)     | + convs. +Params  | 1.74M   | 86% |
| [2.0](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v2.png)       | Deeper. Inception modules. Residual conections. Very cool  |  16.25M  | 90.6% |
|State-of-the-art    | Complex architectures. ImAug. InNorm |  2 - 35 M|  91.5 - 96.5% |

### Batch normalization 
Version__ used to simplify and speed up the process.
See Stanford videos
### Preprocess
### Inizializers
### Regularizations
(thanks to BatchNorm nos e necesita dropout)
See Stanford videos
### Activations
See Stanford videos
### Optimizers
    
## Things yet to be tried

- [x] Mix and add different convolutions -> V1.3
- [x] Normalize inputs -> [get mean and std](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/Utils/preprocess_dataset.py) and [apply preprocess](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/train.py)
- [x] image augmentation
- [ ] Try Batch normalization after Relu instead of before
- [ ] Try not to preprocess (because Batch norm) only x/255 - 0.5 vs Mean substraction 3 channels vs pixel level vs festurewise
- [ ] Compare this inizializers: truncated_normal, lecun_normal, glorot_uniform, VarianceScaling, he_normal, he_uniform
- [ ] Try different regularizations: conv2D:  kernel_regularizer=l2(weight_decay), l1_l2, dropouts (0.2/0.5) and for BatcNorm : gamma_regularizer=l2(weight_decay),    beta_regularizer=l2(weight_decay), 
- [ ] Try different activations: relu, selu, and (advanced_activations: LeakyReLU, PReLU, ThresholdedReLU)
- [ ] Try different optimizers: Compare speed and accuracy: Adam, SGD, RMSprop, Nadam, Adamax
- [ ] Try using other hyperparametres, like weight decay and its different types (use  keras scheduler).
- [x] Try to join image information with other type of information (multimodal)
- [ ] [Importance sampling](http://idiap.ch/~katharas/importance-sampling/)
- [x] Use some residual conections (See DenseNet and Resnet and Inception arquitectures and theri keras implementations in order to learn why and how)

*(Research also theory about it seeing what are avaidable in Keras).

Striding vs pooling: Striding takes less computation. Pooling takes to converge somewhat earlier. Poolings tends to get 1% more accuracy.

## Things to learn
- [ ] [Standford videos course](https://youtu.be/bNb2fEVKeEo?t=1804) 5/16 watched 



## My notes

May be show some kernels in order to see wat is learning?
Try multi-ipunt?(concatenating them in some convos with the same size or seing how)
(concatenations have to be the same size, that's why DenseNet did not use pools and use zeropading-1 in the dense blocks)

Some blogs:

https://blog.keras.io/ 
http://machinelearningmastery.com/blog/ 
https://machinelearning.apple.com/
http://blog.echen.me/ 
https://blogs.nvidia.com/blog/category/deep-learning/ 
http://karpathy.github.io/ 
https://www.blog.google/topics/machine-learning/ 

