# ShatheNet

This subproject is focused on learning more about convolutional neuroanal networks by building my own CNN. Starting from a very simplistic CNN, I'll try to get the higher accuracy on a certain dataset.

## Datasets
[Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html): This is the data-set I am going to work with. 50k training images and 10k test iamges. 10 classes. 32 x 32 pixels although they are upsampled while training.

The other dataset I will work with is [Caltech-256 Object Category Dataset](http://authors.library.caltech.edu/7694/). A set of 256 object categories containing a total of 30607 images. 
Its size is 1GB. I use this dataset instead of Iamgenet due to its size which makes the development and test easier and faster.


## Progression

| Version        | Notes           | Params           | Cifar-10 Accuracy |
| ------------- |:-------------:|:-------------:| -----:|
| [1.0](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_0.png)     | First simple approach. 60 epochs  | 17K   | 59% |
| [1.1](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_1.png)     | First simple approach. Image augmentation. 70 epochs  | 1.25M   | 74% |
| [1.2](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/images/v1_2.png)     | First simple approach. Image augmentation. 70 epochs  | 1.25M   | __% |
|Inception-v3     |  Not finetuned. Image augmentation. 60 epochs |  24 M|  91% |

## Versions
[ShatheNet_v1.0](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#ShatheNet_v1_0)
[ShatheNet_v1.1](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#ShatheNet_v1_1)
[ShatheNet_v1.2](https://github.com/Shathe/DeepLearning/tree/master/ShatheNet/models/ShatheNet.py#ShatheNet_v1_2)
    
## Things yet to be tried

- [x] Mix and add different convolutions -> V1.2
- [ ] Normalize the inputs (Load with numpy so I can use .fit() instead .fitFromFlow()?)
- [ ] image augmentation
- [ ] Add batch normalization*
- [ ] Try different weight inizializations*
- [ ] Try different regularizations(L2, droput, Batch nrom, gradient checking..)*
- [ ] Try different activations*
- [ ] Striding vs pooling
- [ ] Try different optimizers / function losses*
- [ ] Try GlobalAveragePooling2D*
- [ ] Try using other hyperparametres, like weight decay and its different types (use  keras scheduler).
- [ ] Use some residual conections (See DenseNet and Resnet and Inception arquitectures and theri keras implementations in order to learn why and how)

*(Research also theory about it seeing what are avaidable in Keras).



## Things to learn
- [ ] [Standford videos course](https://www.youtube.com/watch?v=vT1JzLTH4G4&t=54s) 2/16 watched 
- [ ] Why do they need non-linear activations functions?
- [ ] Backpropagation how it really works
- [ ] About GRUs LSTM..


## My notes

Use the [functional keras API](https://keras.io/getting-started/functional-api-guide/).
May be show some kernels in order to see wat is learning?
Try multi-ipunt?(concatenating them in some convos with the same size or seing how)
(concatenations have to be the same size, that's why DenseNet did not use pools and use zeropading-1 in the dense blocks)