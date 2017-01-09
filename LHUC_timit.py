import numpy
import theano
from theano import tensor
import cPickle
from io_func.data_io import read_dataset, read_data_args
# import keras
from keras.models import Sequential
from keras import models
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import h5py
from keras.models import load_model
from keras.constraints import maxnorm

train_data_file = 'cv05.pfile.gz,partition=1000m,random=true,stream=false'
valid_data_file = 'valid.pfile.gz,partition=1000m,random=true,stream=false'
test_data_file = 'test.pfile.gz,partition=1000m,random=true,stream=false'
dev_data_file = 'dev.pfile.gz,partition=1000m,random=true,stream=false'

# Reading training dataset
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


model = Sequential()
model.add(Dropout(0.2, input_shape=(440,)))
model.add(Dense(1024, init='uniform', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, init='uniform', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, init='uniform', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, init='uniform', W_constraint=maxnorm(4)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1940, init='uniform'))
model.add(Activation('softmax'))

model.load_weights('weights3.h5')

a = model.get_weights()
# print a[1].shape
# for layers in model.layers:
    # weights = layers.get_weights()

# print weights

n_in = 440
n_hidden = 1024
n_out = 1940

x = tensor.matrix('x')
W1_value = a[0]
b1_value = a[1]
W2_value = a[2]
b2_value = a[3]
W3_value = a[4]
b3_value = a[5]
W4_value = a[6]
b4_value = a[7]
W_out_value = a[8]
b_out_value = a[9]
# print b_out_value.shape
# print W_out_value.shape
l1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_in+n_hidden)),
        high=numpy.sqrt(6./(n_in+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

l2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

l3_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

l4_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

W1 = theano.shared(value=W1_value,
                   name='W1',
                   borrow=True)
b1 = theano.shared(value=b1_value,
                   name='b1',
                   borrow=True)
W2 = theano.shared(value=W2_value,
                   name='W2',
                   borrow=True)
b2 = theano.shared(value=b2_value,
                   name='b2',
                   borrow=True)
W3 = theano.shared(value=W3_value,
                   name='W3',
                   borrow=True)
b3 = theano.shared(value=b3_value,
                   name='b3',
                   borrow=True)
W4 = theano.shared(value=W4_value,
                   name='W4',
                   borrow=True)
b4 = theano.shared(value=b4_value,
                   name='b4',
                   borrow=True)
W_out = theano.shared(value=W_out_value,
                      name='W_out',
                      borrow=True)
b_out = theano.shared(value=b_out_value,
                      name='b_out',
                      borrow=True)
l1 = theano.shared(value=l1_value,
                   name='l1',
                   borrow=True)
l2 = theano.shared(value=l2_value,
                   name='l2',
                   borrow=True)
l3 = theano.shared(value=l3_value,
                   name='l3',
                   borrow=True)
l4 = theano.shared(value=l4_value,
                   name='l4',
                   borrow=True)

y = tensor.lvector('y')

hidden1 = tensor.nnet.relu(tensor.dot(x, W1) + b1)
hidden2 = tensor.nnet.relu(tensor.dot(hidden1, W2) + b2)
hidden3 = tensor.nnet.relu(tensor.dot(hidden2, W3) + b3)
hidden4 = tensor.nnet.relu(tensor.dot(hidden3, W4) + b4)
hidden_out = tensor.dot(hidden4, W_out) + b_out

p_y_given_x = tensor.nnet.softmax(hidden_out)
y_pred = tensor.argmax(p_y_given_x, axis=1)
log_prob = tensor.log(p_y_given_x)
log_likelihood = log_prob[tensor.arange(y.shape[0]), y]
loss = - log_likelihood.mean()

misclass_nb = tensor.neq(y_pred, y)
misclass_rate = misclass_nb.mean()
test_model = theano.function(inputs=[x,y],
                             outputs=misclass_rate)
y_predictor = theano.function(inputs=[x],
                              outputs=[y_pred])
y_predicted = y_predictor(test_set_x)
train_error = test_model(train_set_x, train_set_y)
test_error = test_model(test_set_x, test_set_y)
print ('Train error: %f'% (train_error))
print ('Test error: %f'% (test_error))
# LHUC network
hidden1_l = tensor.nnet.relu(tensor.dot(x, W1) + b1)*2*tensor.nnet.sigmoid(l1)
hidden2_l = tensor.nnet.relu(tensor.dot(hidden1_l, W2) + b2)*2*tensor.nnet.sigmoid(l2)
hidden3_l = tensor.nnet.relu(tensor.dot(hidden2_l, W3) + b3)*2*tensor.nnet.sigmoid(l3)
hidden4_l = tensor.nnet.relu(tensor.dot(hidden3_l, W4) + b4)*2*tensor.nnet.sigmoid(l4)
hidden_out_l = tensor.dot(hidden4_l, W_out) + b_out

p_y_given_x_lhuc = tensor.nnet.softmax(hidden_out_l)
y_pred_lhuc = tensor.argmax(p_y_given_x_lhuc, axis=1)
log_prob_lhuc = tensor.log(p_y_given_x_lhuc)
log_likelihood_lhuc = log_prob_lhuc[tensor.arange(y.shape[0]), y]
loss_lhuc = - log_likelihood_lhuc.mean()

g_l1 = tensor.grad(cost=loss_lhuc, wrt=l1)
g_l2 = tensor.grad(cost=loss_lhuc, wrt=l2)
g_l3 = tensor.grad(cost=loss_lhuc, wrt=l3)
g_l4 = tensor.grad(cost=loss_lhuc, wrt=l4)
learning_rate = numpy.float32(0.13)
new_l1 = l1 - learning_rate*g_l1
new_l2 = l2 - learning_rate*g_l2
new_l3 = l3 - learning_rate*g_l3
new_l4 = l4 - learning_rate*g_l4

update = [(l1, new_l1), (l2, new_l2), (l3, new_l3), (l4, new_l4)]

lhuc_model = theano.function(inputs=[x,y],
                             outputs=loss_lhuc,
                             updates=update)

misclass_nb_lhuc = tensor.neq(y_pred_lhuc, y)
misclass_rate_lhuc = misclass_nb_lhuc.mean()

lhuc_model_test = theano.function(inputs=[x,y],
                                  outputs=misclass_rate_lhuc)

batch_size = 500
n_test_batches = test_set_x.shape[0] // batch_size
n_train_batches = train_set_x.shape[0] // batch_size
n_dev_batches = dev_set_x.shape[0] // batch_size
n_valid_batches = valid_set_x.shape[0] // batch_size

def get_minibatch(i, dataset_x, dataset_y):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    batch_x = dataset_x[start_idx:end_idx]
    batch_y = dataset_y[start_idx:end_idx]
    return batch_x, batch_y

y_predicted_test = y_predictor(test_set_x)
y_predicted_test = numpy.asarray(y_predicted_test)
y_predicted_test = y_predicted_test[0]
"""
n_epoch = 100

epoch = 0
while epoch < n_epoch:
    epoch = epoch+1
    for minibatch_index in range(25):
        minibatch_x, minibatch_y = get_minibatch(minibatch_index, test_set_x, y_predicted_test)
        test_loss = lhuc_model(minibatch_x, minibatch_y)
    lhuc_test_loss = lhuc_model_test(test_set_x, test_set_y)
    print ('lhuc: epoch %d error %f'% (epoch, lhuc_test_loss))

lhuc_weights = {"l1": l1.get_value(), "l2": l2.get_value(), "l3": l3.get_value(), "l4": l4.get_value()}

cPickle.dump(lhuc_weights, open("lhuc_weights.pkl", "wb"))
"""

W_linear_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_in+n_in)),
        high=numpy.sqrt(6./(n_in+n_in)),
        size=(n_in, n_in)
    ),
    dtype=theano.config.floatX
)

b_linear_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_in+n_in)),
        high=numpy.sqrt(6./(n_in+n_in)),
        size=(n_in,)
    ),
    dtype=theano.config.floatX
)

W_linear = theano.shared(value=W_linear_value,
                         name='W_linear',
                         borrow=True)
b_linear = theano.shared(value=b_linear_value,
                         name='b_linear',
                         borrow=True)
hidden_linear = tensor.dot(x, W_linear) + b_linear
hidden1_lin = tensor.nnet.relu(tensor.dot(hidden_linear, W1) + b1)
hidden2_lin = tensor.nnet.relu(tensor.dot(hidden1_lin, W2) + b2)
hidden3_lin = tensor.nnet.relu(tensor.dot(hidden2_lin, W3) + b3)
hidden4_lin = tensor.nnet.relu(tensor.dot(hidden3_lin, W4) + b4)
hidden_out_lin = tensor.dot(hidden4_lin, W_out) + b_out

p_y_given_x_lin = tensor.nnet.softmax(hidden_out_lin)
y_pred_lin = tensor.argmax(p_y_given_x_lin, axis=1)
log_prob_lin = tensor.log(p_y_given_x_lin)
log_likelihood_lin = log_prob_lin[tensor.arange(y.shape[0]), y]
L2_W = tensor.sum(W_linear**2)
L2_b = tensor.sum(b_linear**2)
lambda_1 = 0.01
loss_linear = - log_likelihood_lin.mean() + lambda_1*(L2_b+L2_W)

g_W_linear, g_b_linear = tensor.grad(cost=loss_linear, wrt=[W_linear, b_linear])
new_W_linear = W_linear - learning_rate*g_W_linear
new_b_linear = b_linear - learning_rate*g_b_linear
update = [(W_linear, new_W_linear), (b_linear, new_b_linear)]

linear_model = theano.function(inputs=[x, y],
                               outputs=loss_linear,
                               updates=update)

misclass_nb_lin = tensor.neq(y_pred_lin, y)
misclass_rate_lin = misclass_nb_lin.mean()

linear_model_test = theano.function(inputs=[x, y],
                                    outputs=[misclass_rate_lin])

batch_size = 500
n_epoch = 100

epoch = 0

while epoch < n_epoch:
    epoch = epoch+1
    for minibatch_index in range(25):
        minibatch_x, minibatch_y = get_minibatch(minibatch_index, test_set_x, y_predicted_test)
        minibatch_loss = linear_model(minibatch_x, minibatch_y)
    linear_error = linear_model_test(test_set_x, test_set_y)
    # print linear_error
    linear_error_training = linear_model_test(train_set_x, train_set_y)
    print ('linear Train: %d error %f'% (epoch, linear_error_training[0]*100))
    print ('linear: epoch %d error %f'% (epoch, linear_error[0]*100))

linear_weights = {"W_linear": W_linear.get_value(), "b_linear": b_linear.get_value()}
cPickle.dump(linear_weights, open("linear_weights.pkl", "wb"))

