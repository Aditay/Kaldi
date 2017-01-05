import numpy
import theano
from theano import tensor
import cPickle
from io_func.data_io import read_dataset, read_data_args

dev_data_file = 'dev.pfile.gz,partition=1000m,random=true,stream=false'

# Reading dev dataset

dev_dataset, dev_dataset_args = read_data_args(dev_data_file)
dev, dev_xy, dev_x, dev_y, dev_set_x , dev_set_y = read_dataset(dev_dataset, dev_dataset_args)
dev_set_y = dev_set_y.astype(numpy.int64)
print (numpy.shape(dev_set_x))

# print a

# Preparing neural network model
n_input = 440
n_output = 1940
n_hidden = 2000

x = tensor.dmatrix('x')
# Loading trained neural network model weights

a = cPickle.load(open('trained_model_weights.pkl', 'rb'))
W_h1_value = numpy.asarray(a['W_h1'])
b_h1_value = numpy.asarray(a['b_h1'])
W_h2_value = numpy.asarray(a['W_h2'])
b_h2_value = numpy.asarray(a['b_h2'])
W_h3_value = numpy.asarray(a['W_h3'])
b_h3_value = numpy.asarray(a['b_h3'])
W_h4_value = numpy.asarray(a['W_h4'])
b_h4_value = numpy.asarray(a['b_h4'])
W_out_value = numpy.asarray(a['W_out'])
b_out_value = numpy.asarray(a['b_out'])

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
c = theano.function(inputs=[x],
                    outputs=[log_prob])
d = c(dev_set_x)
d = numpy.asarray(d)
print (d)
print (d.shape)
misclass_nb = tensor.neq(y_predicted, y)
misclass_rate = misclass_nb.mean()

test_model = theano.function(inputs=[x, y],
                             outputs=misclass_rate)
dev_loss = test_model(dev_set_x, dev_set_y)
print ('test loss %f' % (dev_loss*100))
# print (dev_set_y)

# Adaptation is happening here
batch_size = 500
n_dev_batches = dev_set_x.shape[0] // batch_size

def get_minibatch(i, dataset_x, dataset_y):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    batch_x = dataset_x[start_idx:end_idx]
    batch_y = dataset_y[start_idx:end_idx]
    return batch_x, batch_y

# Predicted output
y_predictor = theano.function(inputs=[x],
                              outputs=y_predicted)

y_pred = y_predictor(dev_set_x)

l1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_input+n_hidden)),
        high=numpy.sqrt(6./(n_input+n_hidden)),
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

W_linear_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_input+n_input)),
        high=numpy.sqrt(6./(n_input+n_input)),
        size=(n_input, n_input)

    ),
    dtype=theano.config.floatX
)

b_linear_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_input+n_input)),
        high=numpy.sqrt(6./(n_input+n_input)),
        size=(n_input,)
    ),
    dtype=theano.config.floatX
)

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
W_linear = theano.shared(value=W_linear_value,
                         name='W_linear',
                         borrow=True)
b_linear = theano.shared(value=b_linear_value,
                         name='b_linear',
                         borrow=True)
# Linear Input Layer adaptation model is build here

hidden_linear = tensor.dot(x, W_linear) + b_linear
hidden1_linear = tensor.nnet.relu(tensor.dot(hidden_linear, W_h1) + b_h1)
hidden2_linear = tensor.nnet.relu(tensor.dot(hidden1_linear, W_h2) + b_h2)
hidden3_linear = tensor.nnet.relu(tensor.dot(hidden2_linear, W_h3) + b_h3)
hidden4_linear = tensor.nnet.relu(tensor.dot(hidden3_linear, W_h4) + b_h4)
hidden_out_linear = tensor.dot(hidden4_linear, W_out) + b_out

p_y_given_x_linear = tensor.nnet.softmax(hidden_out_linear)
y_predicted_linear = tensor.argmax(p_y_given_x_linear, axis=1)
log_prob_linear = tensor.log(y_predicted_linear)

a = theano.function(inputs=[x],
                    outputs=[log_prob_linear])
b = a(dev_set_x)
print (b)
b = numpy.asarray(b)

print (b.shape)
# print (log_prob_linear.eval(dev_set_x))
log_likelihood_linear = log_prob_linear[tensor.arange(y.shape[0]), y]
loss_linear = - log_prob_linear.mean()

# Learning rate common for all
learning_rate = numpy.float32(0.13)

# Linear layer training starts here

g_W_linear, g_b_linear = tensor.grad(cost=loss_linear, wrt=[W_linear, b_linear])
new_W_linear = W_linear - learning_rate*g_W_linear
new_b_linear = b_linear - learning_rate*g_b_linear

update = [(W_linear, new_W_linear), (b_linear, new_b_linear)]

linear_model = theano.function(inputs=[x, y],
                               outputs=[loss_linear],
                               updates=update)

misclass_nb_linear = theano.neq(y_predicted_linear, y)
misclass_rate_linear = misclass_nb_linear.mean()

linear_model_test = theano.function(inputs=[x, y],
                                    outputs=misclass_rate_linear)

batch_size = 300
y_pred = numpy.asarray(y_pred)
y_pred = y_pred[0]

n_epoch_linear = 20

epoch = 0

while epoch < n_epoch_linear:
    epoch = epoch + 1
    for minibatch_index in range(20):
        minibatch_x, minibatch_y = get_minibatch(minibatch_index, dev_set_x, y_pred)
        minibatch_avg_index = linear_model(minibatch_x, minibatch_y)
    linear_loss = linear_model_test(dev_set_x, dev_set_y)
    print ('linear epoch %d , test loss linear %f'%(epoch, linear_loss))





