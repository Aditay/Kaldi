import numpy
import theano
from theano import tensor
import cPickle
from io_func.data_io import read_dataset, read_data_args
# data_file = 'cv05.pfile.gz,partition=1000m,random=true,stream=false'
# train_dataset, train_dataset_args = read_data_args(data_file)
# train_set, train_xy, train_x, train_y, x , y = read_dataset(train_dataset, train_dataset_args)
# Address of datasets
train_data_file = 'cv05.pfile.gz,partition=1000m,random=true,stream=false'
valid_data_file = 'valid.pfile.gz,partition=1000m,random=true,stream=false'
test_data_file = 'test.pfile.gz,partition=1000m,random=true,stream=false'

# Reading training dataset
train_dataset, train_dataset_args = read_data_args(train_data_file)
train, train_xy, train_x, train_y, train_set_x , train_set_y = read_dataset(train_dataset, train_dataset_args)

# Reading validation dataset
valid_dataset, valid_dataset_args = read_data_args(valid_data_file)
valid, valid_xy, valid_x, valid_y, valid_set_x, valid_set_y = read_dataset(valid_dataset, valid_dataset_args)

# Reading test dataset
test_dataset, test_data_args = read_data_args(test_data_file)
test, test_xy, test_x, test_y, test_set_x, test_set_y = read_dataset(test_dataset, test_data_args)

print (test_set_y)

# train_set_y = numpy.asarray(train_set_y, dtype=numpy.int64)
# print (train_set_y.dtype)
train_set_y = train_set_y.astype(numpy.int64)
valid_set_y = valid_set_y.astype(numpy.int64)
test_set_y = test_set_y.astype(numpy.int64)

print (train_set_y.dtype)
# print (test_set_x.shape)
# print (train_set_x.shape)
# print (valid_set_x.shape)

# Preparing neural network model
n_input = 440
n_output = 1940
n_hidden = 1024

x = tensor.dmatrix('x')
W_h1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_input+n_hidden)),
        high=numpy.sqrt(6./(n_input+n_hidden)),
        size=(n_input,n_hidden)
    ),
    dtype=theano.config.floatX
)

b_h1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_input+n_hidden)),
        high=numpy.sqrt(6./(n_input+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

W_h2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden, n_hidden)
    ),
    dtype=theano.config.floatX
)

b_h2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

W_h3_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden, n_hidden)
    ),
    dtype=theano.config.floatX
)

b_h3_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

W_h4_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden, n_hidden)
    ),
    dtype=theano.config.floatX
)

b_h4_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_hidden)),
        high=numpy.sqrt(6./(n_hidden+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)

W_out_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_output)),
        high=numpy.sqrt(6./(n_hidden+n_output)),
        size=(n_hidden, n_output)
    ),
    dtype=theano.config.floatX
)

b_out_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_output)),
        high=numpy.sqrt(6./(n_hidden+n_output)),
        size=(n_output,)
    ),
    dtype=theano.config.floatX
)

W_h1 = theano.shared(value=W_h1_value,
                     name='W_h1',
                     borrow=True)
b_h1 = theano.shared(value=b_h1_value,
                     name='b_h1',
                     borrow=True)
W_h2 = theano.shared(value=W_h2_value,
                     name='W_h2',
                     borrow=True)
b_h2 = theano.shared(value=b_h2_value,
                     name='b_h2',
                     borrow=True)
W_h3 = theano.shared(value=W_h3_value,
                     name='W_h3',
                     borrow=True)
b_h3 = theano.shared(value=b_h3_value,
                     name='b_h3',
                     borrow=True)
W_h4 = theano.shared(value=W_h4_value,
                     name='W_h4',
                     borrow=True)
b_h4 = theano.shared(value=b_h4_value,
                     name='b_h4',
                     borrow=True)
W_out = theano.shared(value=W_out_value,
                      name='W_out',
                      borrow=True)

b_out = theano.shared(value=b_out_value,
                      name='b_out',
                      borrow=True)

hidden1 = tensor.nnet.relu(tensor.dot(x, W_h1) + b_h1)
hidden2 = tensor.nnet.relu(tensor.dot(hidden1, W_h2) + b_h2)
hidden3 = tensor.nnet.relu(tensor.dot(hidden2, W_h3) + b_h3)
hidden4 = tensor.nnet.relu(tensor.dot(hidden3, W_h4) + b_h4)
hidden_out = tensor.dot(hidden4, W_out) + b_out

p_y_given_x = tensor.nnet.softmax(hidden_out)
y_predicted = tensor.argmax(p_y_given_x, axis=1)

# Loss function is defined here
y = tensor.lvector('y')
log_prob = tensor.log(p_y_given_x)
log_likelihood = log_prob[tensor.arange(y.shape[0]), y]
loss = - log_likelihood.mean()

# Model Training and cross validation

g_Wh1, g_Wh2, g_Wh3, g_Wh4, g_Wout, g_bh1, g_bh2, g_bh3, g_bh4, g_bout = tensor.grad(cost=loss, wrt=[W_h1, W_h2, W_h3, W_h4, W_out, b_h1, b_h2, b_h3, b_h4, b_out])

learning_rate = numpy.float32(0.13)
new_Wh1 = W_h1 - learning_rate*g_Wh1
new_Wh2 = W_h2 - learning_rate*g_Wh2
new_Wh3 = W_h3 - learning_rate*g_Wh3
new_Wh4 = W_h4 - learning_rate*g_Wh4
new_W_out = W_out - learning_rate*g_Wout
new_b_h1 = b_h1 - learning_rate*g_bh1
new_b_h2 = b_h2 - learning_rate*g_bh2
new_b_h3 = b_h3 - learning_rate*g_bh3
new_b_h4 = b_h4 - learning_rate*g_bh4
new_b_out = b_out - learning_rate*g_bout

update = [(W_h1, new_Wh1), (W_h2, new_Wh2), (W_h3, new_Wh3), (W_h4, new_Wh4), (W_out, new_W_out), (b_h1, new_b_h1), (b_h2, new_b_h2), (b_h3, new_b_h3), (b_h4, new_b_h4), (b_out, new_b_out)]

train_model = theano.function(inputs=[x, y],
                              outputs=[loss],
                              updates=update)

misclass_nb = tensor.neq(y_predicted, y)
misclass_rate = misclass_nb.mean()

test_model = theano.function(inputs=[x, y],
                             outputs=misclass_rate)

batch_size = 500
n_train_batches = train_set_x.shape[0] // batch_size
n_valid_batches = valid_set_x.shape[0] // batch_size
n_test_batches = test_set_x.shape[0] // batch_size

def get_minibatch(i, dataset_x, dataset_y):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    batch_x = dataset_x[start_idx:end_idx]
    batch_y = dataset_y[start_idx:end_idx]
    return (batch_x, batch_y)

n_epoch = 40
# from six.moves import xrange

epoch = 0

while epoch < n_epoch:
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_x, minibatch_y = get_minibatch(minibatch_index, train_set_x, train_set_y)
        minibatch_loss = train_model(minibatch_x, minibatch_y)
    validation_loss = test_model(valid_set_x, valid_set_y)
    print ('epoch %d, validation accuracy %f'% (epoch, validation_loss*100))

# Saving model
weights = {"W_h1": W_h1.get_value(), "W_h2": W_h2.get_value(), "W_h3": W_h3.get_value(), "W_h4": W_h4.get_value(), "W_out": W_out.get_value(), "b_h1": b_h1.get_value(), "b_h2": b_h2.get_value(), "b_h3": b_h3.get_value(), "b_h4": b_h4.get_value(), "b_out": b_out.get_value()}
cPickle.dump(weights, open("trained_model_weights.pkl", "wb"))

# Model testing
test_loss = test_model(test_set_x, test_set_y)
print ('test loss %f' % (test_loss*100))
print (test_set_y)

# output value calculator

# y_predctor = theano.function(inputs=[x],
#                              outputs=[y_predicted])
# y_predicted
