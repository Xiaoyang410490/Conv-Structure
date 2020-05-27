# Conv-Structure
This is a traffic-sign recognition task using convolutional neural network(CNN)
The code is written in Tensorflow 1.9 still using the static map with api like tf.Session() or tf.placeholder(), so it may not be suitable for new tensorflow version like Tensorflow2.1
Basic Net is the baseline model with LeNet structure
LiuNet Test1 implements Leaky-Relu instead of ReLu compared to Basic Net
LiuNet Test1.2 implements ELU activation function instead of Leaky-Relu
LiuNet Test2 tries the small convolutional kern 3*3 instead of 5*5
LiuNet Test3.2 and LiuNet Test3 real utilizes deep neural network replacing the FC-layer with Conv layer
LiuNet Test4 real is the model with best performance implementing Multi-scale feature fusion.
