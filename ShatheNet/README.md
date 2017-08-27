# ShatheNet

This subproject is focused on learning more about convolutional neuroanal networks by building my own CNN. Starting from a very simplistic CNN, I'll try to get the higher accuracy on a certain dataset.

## Datasets
[Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html): This is the data-set I am going to work with. 50k training images and 10k test iamges. 10 classes. 32 x 32 pixels although they are upsampled while training.

The other dataset I will work with is [Caltech-256 Object Category Dataset](http://authors.library.caltech.edu/7694/). A set of 256 object categories containing a total of 30607 images. 
Its size is 1GB. I use this dataset instead of Iamgenet due to its size which makes the development and test easier and faster.


## Progression
### Cifar-10

| Version        | Notes           | Params           | Cifar-10 Accuracy |
| ------------- |:-------------:|:-------------:| -----:|
| [1.0](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_0.png)     | First simple approach   | 17K   | 60% |
| [1.1](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_2.png)     | + convs. +Params. ImAug  | 1.25M   | 76% |
| [1.2](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_1.png)     | + convs. - Params. ImAug. InNorm? | 930K   | 83% |
| [1.3](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_3.png)     | + convs. +Params. ImAug  | 1.74M   | 86% |
|State-of-the-art    | Complex architectures. ImAug. InNorm |  2 - 35 M|  91.5 - 96.5% |

ImAug=Image augmentation
InNorm=Input normalization

### Caltech-256

| Version        | Notes           | Params           | Cifar-10 Accuracy |
| ------------- |:-------------:|:-------------:| -----:|
| [1.3](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_3.png)     |   | 1.74M   |  |
| [2.0]()     |    |  | |
|State-of-the-art    | Complex architectures. ImAug. InNorm |  2 - 35 M|   |

ImAug=Image augmentation
InNorm=Input normalization
## Versions
[ShatheNet_v1.0](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#L10)
[ShatheNet_v1.1](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#L32)
[ShatheNet_v1.2](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#59)
[ShatheNet_v1.3](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#94)
    
## Things yet to be tried

- [x] Mix and add different convolutions -> V1.3
- [x] Normalize inputs -> [get mean and std](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/Utils/preprocess_dataset.py) and [apply preprocess](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/train.py)
- [x] image augmentation
- [ ] Add batch normalization*
- [ ] Try different weight inizializations*
- [ ] Try different regularizations(L2, droput, Batch nrom, gradient checking..)*
- [ ] Try different activations*
- [ ] Mean substraction 3 channels vs pixel level
- [ ] Striding vs pooling on VGG16
- [ ] Try different optimizers / function losses*
- [ ] Try GlobalAveragePooling2D*
- [ ] Try using other hyperparametres, like weight decay and its different types (use  keras scheduler).
- [ ] Try to join image information with other type of information (multimodal)
- [ ] [Importance sampling](http://idiap.ch/~katharas/importance-sampling/)
- [ ] Use some residual conections (See DenseNet and Resnet and Inception arquitectures and theri keras implementations in order to learn why and how)

*(Research also theory about it seeing what are avaidable in Keras).

Striding vs pooling: Striding takes less computation. Pooling takes to converge somewhat earlier. Poolings tends to get 1% more accuracy.

## Things to learn
- [ ] [Standford videos course](https://www.youtube.com/watch?v=vT1JzLTH4G4&t=54s) 4/16 watched 



## My notes

Use the [functional keras API](https://keras.io/getting-started/functional-api-guide/).
May be show some kernels in order to see wat is learning?
Try multi-ipunt?(concatenating them in some convos with the same size or seing how)
(concatenations have to be the same size, that's why DenseNet did not use pools and use zeropading-1 in the dense blocks)