import numpy
import tensorflow as tf
import theano
from theano import tensor
import cPickle
from sklearn.manifold import TSNE
from io_func.data_io import read_dataset, read_data_args
from keras.utils import np_utils
import scipy.io

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

train_data_file = 'train.pfile.gz,partition=1000m,random=true,stream=false'
valid_data_file = 'valid.pfile.gz,partition=1000m,random=true,stream=false'
test_data_file = 'felc0.pfile.gz,partition=1000m,random=true,stream=false'
dev_data_file = 'felc0d.pfile.gz,partition=1000m,random=true,stream=false'

train_dataset, train_dataset_args = read_data_args(train_data_file)
train, train_xy, train_x, train_y, train_set_x , train_set_y = read_dataset(train_dataset, train_dataset_args)

# Reading validation dataset
valid_dataset, valid_dataset_args = read_data_args(valid_data_file)
valid, valid_xy, valid_x, valid_y, valid_set_x, valid_set_y = read_dataset(valid_dataset, valid_dataset_args)

# Reading test dataset
test_dataset, test_data_args = read_data_args(test_data_file)
test, test_xy, test_x, test_y, test_set_x, test_set_y = read_dataset(test_dataset, test_data_args)

# Reading dev dataset
dev_dataset, dev_data_args = read_data_args(dev_data_file)
dev, dev_xy, dev_x, dev_y, dev_set_x, dev_set_y = read_dataset(dev_dataset, dev_data_args)

train_set_y = train_set_y.astype(numpy.int64)
valid_set_y = valid_set_y.astype(numpy.int64)
test_set_y = test_set_y.astype(numpy.int64)
dev_set_y = dev_set_y.astype(numpy.int64)
train_set_y = np_utils.to_categorical(train_set_y, nb_classes=1940)
test_set_y = np_utils.to_categorical(test_set_y, nb_classes=1940)
print test_set_y.shape

mat = scipy.io.loadmat('W1.mat')
# print mat
a = mat['W1']
a = numpy.asarray(a)
# print a.dtype
# W1_value = numpy.asarray(mat['W1'])
W1_value = numpy.asarray(mat['W1'])

mat = scipy.io.loadmat('b1.mat')
b1_value = numpy.asarray(mat['b1'])
# print b1_value.shape
b1_value = b1_value.reshape(1024,)
# print b1_value.shape
mat = scipy.io.loadmat('W2.mat')
W2_value = numpy.asarray(mat['W2'])
# print W2_value.shape
mat = scipy.io.loadmat('b2.mat')
b2_value = numpy.asarray(mat['b2'])
# print b2_value.shape
b2_value = b2_value.reshape(1024,)
mat = scipy.io.loadmat('W3.mat')
W3_value = numpy.asarray(mat['W3'])
# print W3_value.shape
mat = scipy.io.loadmat('b3.mat')
b3_value = numpy.asarray(mat['b3'])
# print b3_value.shape
b3_value = b3_value.reshape(1024,)
mat = scipy.io.loadmat('W4.mat')
W4_value = numpy.asarray(mat['W4'])

mat = scipy.io.loadmat('b4.mat')
b4_value = numpy.asarray(mat['b4'])
# print b4_value.shape
b4_value = b4_value.reshape(1024,)
mat = scipy.io.loadmat('W_out.mat')
W_out_value = numpy.asarray(mat['W_out'])
# print W_out_value.shape
mat = scipy.io.loadmat('b_out.mat')
b_out_value = numpy.asarray(mat['b_out'])
# print b_out_value.shape
b_out_value = b_out_value.reshape(1940,)

import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.neg(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()

batch_size = 40
class LHUC_adversarial(object):
    def __init__(self):
        self._build_model()
    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 440])
        self.Y = tf.placeholder(tf.float32, [None, 1940])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        self.W1 = tf.placeholder(tf.float32, [440, 1024])
        self.b1 = tf.placeholder(tf.float32, [1024])
        self.W2 = tf.placeholder(tf.float32, [1024, 1024])
        self.b2 = tf.placeholder(tf.float32, [1024])
        self.W3 = tf.placeholder(tf.float32, [1024, 1024])
        self.b3 = tf.placeholder(tf.float32, [1024])
        self.W4 = tf.placeholder(tf.float32, [1024, 1024])
        self.b4 = tf.placeholder(tf.float32, [1024])
        self.W_out = tf.placeholder(tf.float32, [1024, 1940])
        self.b_out = tf.placeholder(tf.float32, [1940])
        with tf.variable_scope('feature_extractor'):
            # l_1 = tf.get_variable('l1', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
            # l_2 = tf.get_variable('l2', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
            # l_3 = tf.get_variable('l3', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
            l_1 = bias_variable([1024])
            l_2 = bias_variable([1024])
            l_3 = bias_variable([1024])

            hidden_1 = tf.mul(tf.nn.sigmoid(tf.matmul(self.X, self.W1) + self.b1), 2*tf.nn.sigmoid(l_1))
            hidden_2 = tf.mul(tf.nn.sigmoid(tf.matmul(hidden_1, self.W2) + self.b1), 2*tf.nn.sigmoid(l_2))
            hidden_3 = tf.mul(tf.nn.sigmoid(tf.matmul(hidden_2, self.W3) + self.b3), 2*tf.nn.sigmoid(l_3))
            self.features = hidden_3
        with tf.variable_scope('label_predictor'):
            all_features = lambda : self.features
            source_features = lambda : tf.slice(self.features, [0,0], [batch_size/2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)
            all_labels = lambda : self.Y
            source_labels = lambda : tf.slice(self.Y, [0,0], [batch_size/2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            hidden_4 = tf.nn.sigmoid(tf.matmul(classify_feats, self.W4) + self.b4)
            logits = tf.matmul(hidden_4, self.W_out) + self.b_out
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.classify_labels)

        with tf.variable_scope('domain_predictord'):
            feat = flip_gradient(self.features, self.l)
            W_4d = weight_variable([1024, 1024])
            b_4d = bias_variable([1024])
            W_outd = weight_variable([1024, 2])
            b_outd = bias_variable([2])
            # W_4d = tf.get_variable('W_4d', shape=[1024,1024], initializer=tf.contrib.layers.xavier_initializer())
            # b_4d = tf.get_variable('b_4d', shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
            # W_outd = tf.get_variable('W_outd', shape=[1024, 1940], initializer=tf.contrib.layers.xavier_initializer())
            # b_outd = tf.get_variable('b_outd', shape=[1940], initializer=tf.contrib.layers.xavier_initializer())
            hidden_4d = tf.nn.sigmoid(tf.matmul(feat, W_4d) + b_4d)
            d_logits = tf.matmul(hidden_4d, W_outd) + b_outd
            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(d_logits, self.domain)

graph = tf.get_default_graph()
with graph.as_default():
    model = LHUC_adversarial()
    learning_rate = tf.placeholder(tf.float32, [])
    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    adver_training_ops = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

#     Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_accu = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_accu = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

num_steps = 10000


def train_evaluate(graph, model):
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        gen_source_batch = batch_generator([train_set_x[0:10000], train_set_y[0:10000]], batch_size=batch_size/2)
        gen_target_batch = batch_generator([test_set_x[0:1500], test_set_y[0:1500]], batch_size=batch_size/2)
        domain_labels = numpy.vstack([numpy.tile([1., 0.], [batch_size/2, 1]), numpy.tile([0., 1.],[batch_size/2, 1])])
#         Training loop
        for i in range(num_steps):
            p = float(i)/num_steps
            l = 2./(1.+numpy.exp(-10.*p)) - 1
            lr = 0.01/(1. + 10*p)**0.75
            x0, y0 = gen_source_batch.next()
            x1, y1 = gen_target_batch.next()
            # print y1.shape
            # print y0.shape
            X = numpy.vstack([x0, x1])
            Y = numpy.vstack([y0, y1])
            _, batch_loss, dloss, ploss, d_acc, p_acc = \
            sess.run([adver_training_ops, total_loss, domain_loss, pred_loss, domain_accu, label_accu],
                     feed_dict={model.X:X, model.Y:Y, model.domain: domain_labels, model.train:True, model.l:0.2,
            learning_rate:lr, model.W1: W1_value, model.b1: b1_value, model.W2: W2_value, model.b2:b2_value,
                                model.W3: W3_value, model.b3: b3_value, model.W4: W4_value, model.b4: b4_value,
                                model.W_out: W_out_value, model.b_out: b_out_value})
            print 'step %d loss %f d_acc: %f p_acc %f'%(i,batch_loss, d_acc, p_acc)
        target_accuracy = sess.run(label_accu, feed_dict={model.X: test_set_x, model.Y: test_set_y, model.train:False,
                                                          model.W1: W1_value, model.b1: b1_value,
                                                          model.W2: W2_value, model.b2: b2_value,
                                                          model.W3: W3_value, model.b3: b3_value, model.W4: W4_value,
                                                          model.b4: b4_value,
                                                          model.W_out: W_out_value, model.b_out: b_out_value
                                                          })

        print 'target accuracy :%f'%(target_accuracy)

train_evaluate(graph, model)




