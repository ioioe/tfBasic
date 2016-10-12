## MNIST dataset results (60000 training, 10000 validation/test, image size 28*28*1)
#### CNN

```
3*3*32 conv + relu --> 3*3*32 conv + relu --> 2*2 max pooling --> dropout 0.25 --> flatten --> 128 hidden + relu --> dropout 0.5 --> 10 output + softmax
```

0.9898 accuracy after 12 epochs (128 batch size, adadelta)

#### single row-based LSTM

```
LSTM(128) over (28) rows --> 10 output + softmax
```

0.9852 accuracy after 19 epochs (128 batch size, adam)

#### Hierarchical LSTM 

```
LSTM(128) over (28) single pixels in each row --> LSTM(128) over (28) encoded rows --> 10 output + softmax
```

0.9864 accuracy after 14 epochs (32 batch size, rmsprop)

#### 2 LSTM

```
LSTM(28, return sequence) over (28) rows --> LSTM(128) over (28) encoded rows --> 10 output + softmax
```

0.9850 accuracy after 15 epochs (128 batch size, adam)