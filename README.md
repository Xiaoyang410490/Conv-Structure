# Conv-Structure
This is a traffic-sign recognition on the public data set GTSRB using Convolutional Neural Network (CNN).

The code is still written in the old version Tensorflow: Tensorflow 1.9. So the static map is still used with api like tf.Session() or tf.placeholder(), so it may not be suitable for new tensorflow version like Tensorflow2.1


LiuNet Test1 implements the most basic LeNet/AlexNet architecture. Here the Leaky-ReLU activation function is used, which can be changed to ReLU function or ELU function. 

LiuNet Test2 tries the smaller convolutional kern 3*3 instead of 5*5. And this is a VGG like network.

LiuNet Test2 real adds some tricks like Batch Normalization and Dropout. 

LiuNet Test3.2 and LiuNet Test3 real utilizes deep neural network replacing the FC-layer with Conv layer
LiuNet Test4 real is the model with best performance implementing Multi-scale feature fusion.
