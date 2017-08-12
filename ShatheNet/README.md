# ShatheNet

This subproject is focused on learning more about convolutional neuroanal networks by building my own CNN. Starting from a very simplistic CNN, I'll try to get the higher accuracy on a certain dataset.

## Dataset

[Caltech-256 Object Category Dataset](http://authors.library.caltech.edu/7694/). A set of 256 object categories containing a total of 30607 images. 
Its size is 1GB. I use this dataset instead of Iamgenet due to its size which makes the development and test easier and faster.


## Progression

| Version        | Notes           | Accuracy  |
| ------------- |:-------------:| -----:|
| [1.0]()referenciar commit*      | First simple approach. [arquitecture]()refeernce image* | 0.0 |
|       |  | 0.0 |
|Inception-v3     |  Not finetuned. 224x224 input. Image augmentation | 0.0 |
|Resnet-50      | Not finetuned | 0.0 |
## Versions
1.0: Explicar que tiene*
## Things yet to be tried

- [ ] Mix and add different convolutions
- [ ] Normalize the inputs (Load with numpy so I can use .fit() instead .fitFromFlow()?)
- [ ] image augmentation
- [ ] Add batch normalization*
- [ ] Try different weight inizializations*
- [ ] Try different regularizations(L2, droput, Batch nrom, gradient checking..)*
- [ ] Try different activations*
- [ ] Try different optimizers / function losses*
- [ ] Try GlobalAveragePooling2D*
- [ ] Try using other hyperparametres, like weight decay and its different types (use  keras scheduler).
- [ ] Use some residual conections (See DenseNet and Resnet and Inception arquitectures and theri keras implementations in order to learn why and how)

*(Research also theory about it seeing what are avaidable in Keras).



## Things to learn
- [ ] Why do they need non-linear activations functions?
- [ ] Backpropagation how it really works
- [ ] About GRUs LSTM..


## My notes

Use the [functional keras API](https://keras.io/getting-started/functional-api-guide/).
Start using previous projects files.
Divide the project into packages and classes.
May be show some kernels in order to see wat is learning?
Try multi-ipunt?(concatenating them in some convos with the same size or seing how)
(concatenations have to be the same size, that's why DenseNet did not use pools and use zeropading-1 in the dense blocks)