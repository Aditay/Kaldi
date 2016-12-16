# Loading data

import os
import requests
import gzip
import six
from six.moves import cPickle

if not os.path.exists('mnist.pkl.gz'):
    r = requests.get('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    with open('mnist.pkl.gz', 'wb') as data_file:
        data_file.write(r.content)

with gzip.open('mnist.pkl.gz', 'rb') as data_file:
    if six.PY3:
        train_set, valid_set, test_set = cPickle.load(data_file, encoding= 'latin1')
    else:
        train_set, valid_set, test_set = cPickle.load(data_file)

train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set
test_set_x, test_set_y = test_set

# Model is made here

import numpy
import theano
from theano import tensor

n_in = 28*28
n_out = 10
n_hidden = 400

x = tensor.matrix('x')
W_hidden_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_in + n_hidden)),
        high=numpy.sqrt(6./(n_in + n_hidden)),
        size=(n_in, n_hidden)
    ),
    dtype=theano.config.floatX
)

W_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_out)),
        high=numpy.sqrt(6./(n_hidden+n_out)),
        size=(n_hidden, n_out)
    ),
    dtype=theano.config.floatX
)

b_hidden_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_in+n_hidden)),
        high=numpy.sqrt(6./(n_in+n_hidden)),
        size=(n_hidden,)
    ),
    dtype=theano.config.floatX
)
b_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden+n_out)),
        high=numpy.sqrt(6./(n_hidden+n_out)),
        size=(n_out,)
    ),
    dtype=theano.config.floatX
)
W_hidden = theano.shared(value=W_hidden_value,
                         name='W_hidden',
                         borrow=True)
b_hidden = theano.shared(value=b_hidden_value,
                         name='b_hidden',
                         borrow=True)
W = theano.shared(value=W_value,
                  name='W',
                  borrow=True)
b = theano.shared(value=b_value,
                  name='b',
                  borrow=True)
hidden_out = tensor.nnet.relu(tensor.dot(x, W_hidden) + b_hidden)
p_y_given_x = tensor.nnet.softmax(tensor.dot(hidden_out, W) + b)
y_pred = tensor.argmax(p_y_given_x, axis=1)

# Now the loss function is defined

y = tensor.lvector('y')
log_prob = tensor.log(p_y_given_x)
log_likelihood = log_prob[tensor.arange(y.shape[0]), y]
loss = - log_likelihood.mean()

# Training starts here

g_W_hidden, g_b_hidden, g_W, g_b = tensor.grad(cost=loss, wrt=[W_hidden, b_hidden, W, b])

learning_rate = numpy.float32(0.13)
new_W_hidden = W_hidden - learning_rate*g_W_hidden
new_b_hidden = b_hidden - learning_rate*g_b_hidden
new_W = W - learning_rate*g_W
new_b = b - learning_rate*g_b

update = [(W_hidden, new_W_hidden), (b_hidden, new_b_hidden), (W, new_W), (b, new_b)]

train_model = theano.function(inputs=[x, y],
                              outputs=loss,
                              updates=update)

misclass_nb = tensor.neq(y_pred, y)
misclass_rate = misclass_nb.mean()

test_model = theano.function(inputs=[x, y],
                             outputs=misclass_rate)

# Training model

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

## Early stopping parameters
# maximum number of epochs
n_epochs = 1000
# look at this many examples regardless
patience = 5000
# wait this much longer when a new best is found
patience_increase = 2
# a relative improvement of this much is considered significant
improvement_threshold = 0.995

# go through this many minibatches before checking the network on the validation set;
# in this case we check every epoch
validation_frequency = min(n_train_batches, patience / 2)


import timeit
from six.moves import xrange

best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

while(epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_x, minibatch_y = get_minibatch(minibatch_index, train_set_x, train_set_y)
        minibatch_avg_index = train_model(minibatch_x, minibatch_y)

        iteration = (epoch - 1)*n_train_batches + minibatch_index
        if (iteration + 1) % validation_frequency == 0:
            validation_loss = []
            for i in xrange(n_valid_batches):
                valid_xi, valid_yi = get_minibatch(i, valid_set_x, valid_set_y)
                validation_loss.append((test_model(valid_xi, valid_yi)))


            # get_variables = theano.function(inputs=[x, y], outputs=[g_W_hidden, g_b_hidden, g_W, g_b])
            # [W_hidden_value, b_hidden_value, W_value, b_value] = get_variables(valid_set_x, valid_set_y)
            # print ('g_w_hidden %f, g_b_hidden %f, g_W %f, g_b %f' % (W_hidden_value, b_hidden_value, W_value, b_value))
            this_validation_loss = numpy.mean(validation_loss)
            print('epoch %i, minibatch %i/%i, validation error %f '%
                  (epoch,
                   minibatch_index+1,
                   n_train_batches,
                   this_validation_loss*100.))

            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss*improvement_threshold:
                    patience = max(patience, iteration*patience_increase)

                best_validation_loss = this_validation_loss

                test_loss = []
                for i in xrange(n_test_batches):
                    test_xi, test_yi = get_minibatch(i, test_set_x, test_set_y)
                    test_loss.append(test_model(test_xi, test_yi))
                test_score = numpy.mean(test_loss)
                print ('    epoch %i, minibatch %i/%i, test error of best model %f' %
                       (epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score*100.))
                numpy.savez('best_model.npz', W_hidden=W_hidden.get_value(), b_hidden=b_hidden.get_value(), W=W.get_value(), b=b.get_value())
        if patience <= iteration:
            done_looping = True
            break

end_time = timeit.default_timer()
print ('Optimization complete with best validation score of %f %%,' 'with test performance %f %%' % (best_validation_loss*100, test_score*100))
print ('the code ran for %d epochs, with %f epochs/sec' % (epoch, 1.*epoch/(end_time-start_time)))


# from theano.printing import pydotprint
# pydotprint(train_model, outfile='pydotprint_f.png')
# from IPython.display import Image
# Image('pydotprint_f.png', width=1000)
# %matplotlib inline
# import matplotlib.pyplot as plt
# from utils import tile_raster_images

# plt.clf()
# plt.gcf().set_size_inches(15, 10)
# plot_data = tile_raster_images(W.get_value(borrow=True).T,
                               # img_shape=(28,28), tile_shape=(2, 5), tile_spacing=(1,1))
# plt.imshow(plot_data, cmap='Greys', interpolation= 'none')
# plt.axis('off')