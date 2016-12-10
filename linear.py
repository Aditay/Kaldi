from __future__ import print_function
__docformat__ = 'restructuredtext en'

import gzip
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import gc
for i in range(3): gc.collect()
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

class Model(object):
    def __init__(self, input, n_in,n_hidden, n_out):
      self.W_linear = theano.shared(
          value=numpy.zeros(
              (n_in, n_in),
              dtype=theano.config.floatX
          ),
          name= 'W_linear',
          borrow = True
      )
      self.b_linear = theano.shared(
          value=numpy.zeros(
              (n_in,),
              dtype=theano.config.floatX
          ),
          name='b_linear',
          borrow = True
      )
      self.W_h = theano.shared(
          value=numpy.zeros(
              (n_in,n_hidden),
              dtype=theano.config.floatX
          ),
          name='W_h',
          borrow= True
      )
      self.b_h = theano.shared(
          value=numpy.zeros(
              (n_hidden,),
              dtype=theano.config.floatX
          ),
          name='b_h',
          borrow = True
      )
      self.W = theano.shared(
          value=numpy.zeros(
              (n_hidden,n_out),
              dtype=theano.config.floatX
          ),
          name='W',
          borrow= True
      )
      self.b = theano.shared(
          value=numpy.zeros(
              (n_out,),
              dtype=theano.config.floatX
          ),
          name='b',
          borrow=True
      )
    #   Hidden Layer output
      hidden_layer = T.nnet.relu(T.dot(input,self.W_h)+self.b_h)
    #   Single layer network without linear layer at the input
      
      self.p_y_given_x = T.nnet.softmax(T.dot(hidden_layer,self.W)+ self.b)
    #   Predicted labels
      self.y_pred = T.argmax(self.p_y_given_x, axis=1)
    #   Linear layer is added at the input
      linear_hidden_layer = T.dot(input,self.W_linear) + self.b_linear
      hidden_layer_linear_network = T.nnet.relu(T.dot(linear_hidden_layer,self.W_h)+self.b_h)
    #   combined network outputs
      self.p_y_given_x_linear_network = T.nnet.softmax(T.dot(linear_hidden_layer,self.W) + self.b)
      self.y_pred_linear_net = T.argmax(self.p_y_given_x_linear_network,axis= 1)
    #   Keep track of model input
      self.input = input
    
    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def negative_log_likelihood_linear_net(self, y):
        return -T.mean(T.log(self.p_y_given_x_linear_network)[T.arange(y.shape[0]),y])
    
    def error(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError() 

    def error_linear_net(self, y):
        if y.ndim != self.y_pred_linear_net.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred_linear_net.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred_linear_net, y))
        else:
            raise NotImplementedError() 

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
 	test_setx, test_sety = test_set    
 	# test_setx_noisy = test_setx + 0.2*numpy.random.random_sample(numpy.shape(test_setx))
    # test_setx_noisy = test_setx + numpy.random.normal(0.5,0.4)
    # plt.imshow(numpy.reshape(test_setx[1],[28,28]))
    # plt.gray()
    # plt.show()
    test_setx_noisy = test_setx
    [r, c] = numpy.shape(test_setx)
    std = 0.4
    for i in range(r):
        for j in range(c):
            test_setx_noisy[i][j] = test_setx_noisy[i][j] + numpy.random.normal(test_setx[i][j],std)
    '''
    [r,c] = numpy.shape(test_setx)
    prob = 0.1
    test_setx_noisy = test_setx
    thresh = 1 - prob
    for i in range(r):
        for j in range(c):
            rnd = numpy.random.random()
            if rnd < prob:
                test_setx_noisy[i][j] = 0
            elif rnd > thresh:
                test_setx_noisy[i][j] = 1
    '''
    '''
    plt.imshow(numpy.reshape(test_setx_noisy[1],[28,28]))
    plt.gray()
    plt.show()
    '''        
    test_set = (test_setx_noisy, test_sety) 
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    #   self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))  
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def sgd_optimization(learning_rate = 0.13, n_epochs = 1000,
                     dataset='mnist.pkl.gz',
                     batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    adap_batch_size = 400
    n_test_batches_adap = test_set_x.get_value(borrow=True).shape[0] // adap_batch_size
    print('....building the model')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = Model(input=x, n_in = 28 * 28,n_hidden= 500, n_out = 10)
    cost_model = classifier.negative_log_likelihood(y)
    g_W = T.grad(cost= cost_model, wrt = classifier.W)
    g_b = T.grad(cost= cost_model, wrt = classifier.b)
    g_W_h = T.grad(cost= cost_model, wrt = classifier.W_h)
    g_b_h = T.grad(cost= cost_model, wrt= classifier.b_h)

    updates_model = [(classifier.W, classifier.W - learning_rate*g_W),
                     (classifier.b, classifier.b - learning_rate*g_b),
                     (classifier.W_h, classifier.W_h - learning_rate*g_W_h),
                     (classifier.b_h, classifier.b_h - learning_rate*g_b_h)]
    
    train_model = theano.function(
        inputs=[index],
        outputs=cost_model,
        updates = updates_model,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.error(y)],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }

    )
    y_predictor = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred,
        # givens={
            # x: test_set_x[index:],
            # y: test_set_y[index:]
        # }

    )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.error(y),
        givens={
            x: valid_set_x[index* batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    

    patience = 5000
    test_score = 0
    validation_freq = min(n_train_batches, patience // 2)
    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1)* n_train_batches + minibatch_index
            if (iter + 1)%validation_freq == 0:
                n = 0
                loss = 0
                for i in range(n_valid_batches):
                    n = n + 1
                    validate_losses = validate_model(i)
                    loss = loss + validate_losses
                n2 = 0
                lossT = 0
                for i in range(n_test_batches):
                    n2 = n2 + 1
                    [test_losses] = test_model(i)
                    lossT = lossT + test_losses
                    # testLoss = lossT/n2
                this_valid_loss = loss / n
                this_test_loss = loss / n
                print('epoch %i, minibatch %i/%i, validation loss = %f, test loss= %f'%(epoch, minibatch_index,n_train_batches, this_valid_loss*100, this_test_loss*100))
        if patience <= iter:
            done_looping = True
            break
    test_set_x_val = test_set_x.get_value()
    y_predicted = y_predictor(test_set_x_val[:])
    
    # Training the linear layer

    costLinear = classifier.negative_log_likelihood_linear_net(y_predicted)
    g_W_lin = T.grad(cost=costLinear, wrt= classifier.W_linear)
    g_b_lin = T.grad(cost=costLinear, wrt= classifier.b_linear)
    updates_model_lin = [(classifier.W_linear, classifier.W_linear - learning_rate* g_W_lin),
                         (classifier.b_linear, classifier.b_linear - learning_rate*g_b_lin)]
    combined_model = theano.function(
        inputs= [index],
        outputs=[costLinear],
        updates= updates_model_lin,
        givens={
            x: test_set_x[index * adap_batch_size: (index + 1) * adap_batch_size],
            y: y_predicted[index * adap_batch_size: (index + 1) * adap_batch_size] 
        }
    )

    combined_model_test = theano.function(
        inputs = [index],
        outputs=classifier.error_linear_net(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    
    fepoch = 0
    n_fepoch = 50
    while(fepoch < n_fepoch):
        fepoch = fepoch + 1
        ferror = 0
        n3 = 0
        for mn in range(33):
            n3 = n3 +1 
            l_error = combined_model_test(mn)
            ferror = ferror + l_error
        for minibatch_index in range(10):
            l_error2 = combined_model(minibatch_index)
        this_lin_error = ferror/n3
        print('epoch %i, test error after linear layer %f'% (fepoch, this_lin_error*100))

# def predict():

    # predict

if __name__ == '__main__':
    sgd_optimization()
