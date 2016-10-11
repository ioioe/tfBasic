## MNIST dataset results (60000 training, 10000 validation/test)
#### CNN
3*3*32 conv + relu --> 3*3*32 conv + relu --> 2*2 max pooling --> dropout 0.25 --> flatten --> 128 hidden + relu --> dropout 0.5 --> 10 output + softmax

0.9898 accuracy after 12 epochs
