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

train_set_y = train_set_y.astype(numpy.int64)
valid_set_y = valid_set_y.astype(numpy.int64)
test_set_y = test_set_y.astype(numpy.int64)

n_input = 440
n_output = 1940
n_hidden = 1024

x = tensor.dmatrix('x')
W_h1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_input+n_hidden)),
        high=numpy.sqrt(6./(n_input+n_hidden)),
        size=(n_input, n_hidden)
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

