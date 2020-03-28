import pickle
import numpy as np
import tensorflow as tf
import time
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/valid.p'

with open(training_file, mode='rb') as f: train = pickle.load(f)
with open(testing_file, mode='rb') as f: test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
# read the file

n_train = X_train.shape[0]
# TODO: Number of testing examples.
n_test = X_test.shape[0]
# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

X_train_rgb = X_train
X_train_gry = np.sum(X_train / 3, axis=3, keepdims=True)
X_test_rgb = X_test
X_test_gry = np.sum(X_test / 3, axis=3, keepdims=True)
# gry image
X_train_normalized = (X_train_gry - 128.) / 128.
X_test_normalized = (X_test_gry - 128.) / 128.
# normalization



def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def generatebatch(X, Y, n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = np.array(X[start:end])
        batch_ys = np.array(Y[start:end])
        yield batch_xs, batch_ys



x = tf.placeholder("float", shape=[None, 32, 32, 1])
y_ = tf.placeholder("int32", shape=[None])
ys = tf.one_hot(y_, depth=43, on_value=1., off_value=0., axis=-1)

W_conv1 = weight_variable([3, 3, 1, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.elu(conv2d(x, W_conv1) + b_conv1)
W_conv12 = weight_variable([3, 3, 6, 6])
b_conv12 = bias_variable([6])
h_conv12 = tf.nn.elu(conv2d(h_conv1, W_conv12) + b_conv12)
h_pool1 = max_pool_2x2(h_conv12)

W_conv2 = weight_variable([3, 3, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
W_conv22 = weight_variable([3, 3, 16, 16])
b_conv22 = bias_variable([16])
h_conv22 = tf.nn.elu(conv2d(h_conv2, W_conv22) + b_conv22)

W_conv3 = weight_variable([3, 3, 16, 16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.elu(conv2d(h_conv22, W_conv3) + b_conv3)
W_conv32 = weight_variable([3, 3, 16, 16])
b_conv32 = bias_variable([16])
h_conv32 = tf.nn.elu(conv2d(h_conv3, W_conv32) + b_conv32)
h_pool2 = max_pool_2x2(h_conv32)
h_pool2_flat = tf.reshape(h_pool2, [-1, 144])
keep_prob = tf.placeholder("float32")
h_pool2_flat = tf.nn.dropout(h_pool2_flat,keep_prob)

W_fc2 = weight_variable([144, 84])
b_fc2 = bias_variable([84])
h_fc2 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([84, 43])
b_fc3 = bias_variable([43])
logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

y_pred = tf.arg_max(logits, 1)
bool_pred = tf.equal(tf.arg_max(ys, 1), y_pred)
accuracy = tf.reduce_mean(tf.cast(bool_pred, "float"))


training_epochs = 100
batch_size = 89

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    X_train_normalized, y_train = shuffle(X_train_normalized, y_train)
    for batch_xs, batch_ys in generatebatch(X_train_normalized, y_train, n_train, batch_size):
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if (epoch % 1 == 0):
        res = sess.run(accuracy, feed_dict={x: X_train_normalized, y_: y_train, keep_prob: 1.0})
        print(epoch, res)



a=time.time()
res_ypred = y_pred.eval(feed_dict={x: X_test_normalized, y_: y_test, keep_prob: 1.0}).flatten()
b=time.time()
Zeit=b-a
print(Zeit)

print(accuracy_score(y_test, res_ypred.reshape(-1, 1)))