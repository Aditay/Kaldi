# -*- coding: utf-8 -*-
# import numpy as
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import gzip
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import gc
for i in range(3): gc.collect()
import matplotlib.pyplot as plt
# import numpy as np

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.W_linear = theano.shared(
            value=numpy.zeros(
                (n_in,n_in),
                dtype=theano.config.floatX
            ),
            name='W_linear',
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
        self.L = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='L',
            borrow=True
        )
        self.p_y_given_x_lhuc = T.nnet.softmax(T.dot(input, self.W)*2*T.nnet.sigmoid(self.L) + self.b)
        self.p_y_given_x_lhuc_linear = T.nnet.softmax(T.dot((T.dot(input,self.W_linear)+self.b_linear),self.W)*2*T.nnet.sigmoid(self.L)+self.b)
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred_lhuc = T.argmax(self.p_y_given_x_lhuc, axis=1)
        self.y_pred_lhuc_linear = T.argmax(self.p_y_given_x_lhuc_linear, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params_lhuc = self.L
        self.params_linear = [self.W_linear, self.b_linear]
        # self.W2 = theano.shared(
            # value=numpy.zeros(
                # (n_hidden, n_out),
                # dtype=theano.config.floatX
            # ),
            # name='W2',
            # borrow=True
        # )

        # self.b2 = theano.shared(
            # value=numpy.zeros(
                # (n_out,),
                # dtype=theano.config.floatX
            # ),
            # name='b2',
            # borrow=True
        # )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        # self.h1 = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        linearLayer = T.dot(input,self.W_linear) + self.b_linear
        self.p_y_given_x_linear = T.nnet.softmax(T.dot(linearLayer,self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_pred_linear = T.argmax(self.p_y_given_x_linear)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2
    
    
    def costLHUC(self,y):
    	return -T.mean(T.log(self.p_y_given_x_lhuc)[T.arange(y.shape[0]),y])
    def costLinear(self,y):
        return -T.mean(T.log(self.p_y_given_x_linear)[T.arange(y.shape[0]),y])
    def costLHUC_Linear(self,y):
        return -T.mean(T.log(self.p_y_given_x_lhuc_linear)[T.arange(y.shape[0]),y])
    def errorsLHUC(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred_lhuc.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred_lhuc.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred_lhuc, y))
        else:
            raise NotImplementedError()
    def errorsLinear(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        # numpy.shape(self.y_pred)
        # numpy.shape(y)
        '''
        if y.ndim != self.y_pred_linear.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred_linear.type)
            )
        # check if y is of the correct datatype
        '''
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred_linear, y))
        else:
            raise NotImplementedError()
    
    def errorsLHUC_linear(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred_lhuc_linear.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred_lhuc_linear.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred_lhuc_linear, y))
        else:
            raise NotImplementedError()

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
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
    # def addLHUC(self):

        # self.L = theano.shared(
            # value=numpy.zeros(
                # (n_out,),
                # dtype=theano.config.floatX
            # ),
            # name='L',
            # borrow=True
        # )
        # self.p_y_given_x_lhuc = T.nnet.softmax(T.dot(input, self.W)*2*T.nnet.sigmoid(L) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # self.y_pred_lhuc = T.argmax(self.p_y_given_x_lhuc, axis=1)
        # end-snippet-1

        # parameters of the model
        # self.params_lhuc = self.L

        # initialize the biases b as a vector of n_out 0s
        # self.b = theano.shared(
            # value=numpy.zeros(
                # (n_out,),
                # dtype=theano.config.floatX
            # ),
            # name='b',
            # borrow=True
        # )

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
    std = 0.8
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


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    # shape = numpy.shape(test_set_x)
    # r = shape[0]
    # c = shape[1]
    # print(r)
    # print(c)
    # test_set_x = test_set_x + numpy.random.randn(r,c)
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    lhuc_batch_size = 300
    n_test_batches_lhuc = test_set_x.get_value(borrow=True).shape[0] // lhuc_batch_size
    # linear_batch_size = 300
    # n_test_batches
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    g_L = T.grad(cost=classifier.costLHUC(y), wrt = classifier.L)
    updatesLHUC = [(classifier.L, classifier.L - learning_rate * g_L)]
    g_W_linear = T.grad(cost=classifier.costLinear(y), wrt=classifier.W_linear)
    g_b_linear = T.grad(cost=classifier.costLinear(y), wrt=classifier.b_linear)
    g_W_lhuc_linear = T.grad(cost=classifier.costLHUC_Linear(y), wrt = classifier.W_linear)
    g_b_lhuc_linear = T.grad(cost=classifier.costLHUC_Linear(y), wrt=classifier.b_linear)
    g_L_lhuc_linear = T.grad(cost=classifier.costLHUC_Linear(y), wrt=classifier.L)
    updatesLinear = [(classifier.W_linear,classifier.W_linear - learning_rate*g_W_linear),
                    (classifier.b_linear, classifier.b_linear - learning_rate*g_b_linear)
                    ]
    updatesLHUC_Linear = [(classifier.W_linear, classifier.W_linear - learning_rate*g_W_lhuc_linear),
                        (classifier.b_linear, classifier.b_linear - learning_rate*g_b_lhuc_linear),
                        (classifier.L, classifier.L - learning_rate*g_L_lhuc_linear)

                        ]                
    test_model = theano.function(
        inputs=[index],
        updates = updatesLHUC,
        outputs=classifier.errorsLHUC(y),
        givens={
            x: test_set_x[index * lhuc_batch_size: (index + 1) * lhuc_batch_size],
            y: test_set_y[index * lhuc_batch_size: (index + 1) * lhuc_batch_size]
        }
    )
    test_model3 = theano.function(
        inputs=[index],
        # updates = updatesLHUC,
        outputs=classifier.errorsLHUC(y),
        givens={
            x: test_set_x[index * lhuc_batch_size: (index + 1) * lhuc_batch_size],
            y: test_set_y[index * lhuc_batch_size: (index + 1) * lhuc_batch_size]
        }
    )

    # updatesLHUC = [(classifier.L, classifier.L - learning_rate * g_L)]
    # get_weights = theano.function(
        # inputs = [0],
        # outputs= classifier.W
    # )
    test_model2 = theano.function(
        inputs=[index],
        # updates = updatesLHUC,
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    test_model4 = theano.function(
        inputs=[index],
        updates= updatesLinear,
        outputs=classifier.errorsLinear(y),
        givens={
            x: test_set_x[index * lhuc_batch_size: (index + 1) * lhuc_batch_size],
            y: test_set_y[index * lhuc_batch_size: (index + 1) * lhuc_batch_size]
        }
    )

    test_model5 = theano.function(
        inputs=[index],
        # updates = updatesLHUC,
        outputs=classifier.errorsLinear(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model6 = theano.function(
        inputs=[index],
        updates= updatesLHUC_Linear,
        outputs=classifier.errorsLHUC_linear(y),
        givens={
            x: test_set_x[index * lhuc_batch_size: (index + 1) * lhuc_batch_size],
            y: test_set_y[index * lhuc_batch_size: (index + 1) * lhuc_batch_size]
        }
    )

    test_model7 = theano.function(
        inputs=[index],
        # updates = updatesLHUC,
        outputs=classifier.errorsLHUC_linear(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # '''
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # validate_model = theano.function( inputs= )
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    # g_W2 = T.grad(cost=cost, wrt=classifier.W2)
    # g_b2 = T.grad(cost=cost, wrt=classifier.b2)
    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
               # (classifier.W2, classifier.W2 - learning_rate * g_W2),
               # (classifier.b2, classifier.b2 - learning_rate * g_b2)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    # batch_size = 500

    # n_epochs = 50
    # epoch = 0
    # n_train_batches = 50000/500 
    # # while(epoch < n_epochs):
    # 	# epoch =epoch + 1
    # 	for index in xrange(n_train_batches):
    # 		minibatch_avg_cost = train_model(index)
    # 		# iter = (epoch - 1) * n_train_batches + minibatch_index
    # 		validation_losses = [validate_model(i)
    # 							 for i in range(n_valid_batches)]
    # 		this_validation_loss = numpy.mean(validation_losses)
    # 		print('epoch %i, validation error %f'% epoch, this_validation_loss*100)
    patience = 5000
    validation_freq = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_score = 0
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch +1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # print ('%f'% minibatch_avg_cost)
            iter = (epoch -1)* n_train_batches + minibatch_index
            # validation_loss = validate_model(mini)
            # '''
            if(iter +1)%validation_freq == 0:
                # validation_losses = [validate_model(i)
                                    #  for i in range(n_valid_batches)]
                # this_validation_loss = numpy.mean(validation_losses)
                # validation_losses = numpy.zeros([1,1])
                n = 0
                loss = 0
                for i in range(n_valid_batches):
                    n = n +1
                    validate_losses = validate_model(i)
                    loss = loss + validate_losses
                    # print ('%f' % validate_loss)
                    # print numpy.shape(validate_loss)
                    # print numpy.shape(validation_losses)
                    # print validate_loss
                    # numpy.append(validation_losses, validate_loss)
                    # print ('%f' %validation_losses)
                n2 = 0
                lossT = 0
                for i in range(n_test_batches):
                    n2 = n2 + 1
                    test_losses = test_model2(i)
                    lossT = lossT + test_losses
                    testLoss = lossT/n
                this_validation_losses = loss / n
                this_test_losses = lossT / n2
                print ('epoch %i, minibatch %i/%i, validation error %f, test error %f' % (epoch, minibatch_index +1, n_train_batches, this_validation_losses*100, this_test_losses*100))
        if patience <= iter:
            done_looping = True
            break
    fepoch = 0
    n_fepoch = 50
    while(fepoch < n_fepoch):
        fepoch = fepoch + 1
        # print ('%i' % (n_test_batches_lhuc))
        ferror = 0
        n3 = 0
        for mn in range(33):
            n3 = n3 + 1
            lhucError2 = test_model3(mn)
            ferror = ferror + lhucError2
        for minibatch_index_test in range(5):
           lhucError = test_model(minibatch_index_test)        
        this_lhuc_loss = ferror/n3
        print('epoch %i, test error after lhuc %f' % (fepoch, this_lhuc_loss*100))
    # W = classifier.W
    # print (W)
    # print(classifier.W)
    # get_weights = theano.function(inputs= [None], outputs=[classifier.W])
    # W = get_weights()
    # print (W)
        # W = get_weights(0)
        # print (W)
        # print ('weights %f %f %f %f %f %f %f %f ' %(W[0][1], W[3][4], W[4][4],W[350][6], W[28][28], W[63][9], W[84][7], W[8][8]))
    # n = 0
    # lossT = 0
    # for i in range(n_test_batches):
        # n = n + 1
        # test_losses = test_model2(i)
        # lossT = lossT + test_losses
    # testLoss = lossT/n
    # print 
            # '''
    fepoch = 0
    n_fepoch = 50
    while(fepoch < n_fepoch):
        fepoch = fepoch + 1
        ferror = 0
        n3 = 0
        for mn in range(33):
            n3 = n3 + 1
            linearError2 = test_model5(mn)
            ferror = ferror + linearError2
        for minibatch_index_test in range(5):
            linearError = test_model4(minibatch_index_test)
        this_linear_loss = ferror/n3
        print('epoch %i, test error after linear layer %f' % (fepoch,this_linear_loss*100))
    fepoch = 0
    n_fepoch = 50
    while(fepoch < n_fepoch):
        fepoch = fepoch + 1
        ferror = 0 
        n3 = 0
        for mn in range(33):
            n3 = n3 + 1
            linear_lhuc_error2 = test_model7(mn)
            ferror = ferror + linear_lhuc_error2
        for minibatch_index_test in range(5):
            linear_lhuc_Error = test_model6(minibatch_index_test)
        this_lhuc_linear = ferror/n3
        print('epoch %i, test error after both lhuc and linear %f'% (fepoch,this_lhuc_linear*100))



    
    '''
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    '''
'''
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
'''
    # epoch = 0
    # while (epoch < 10):
    	# epoch = epoch + 1
    	# for minibatch_index in range(n_test_batches):
    		# minibatch_avg_cost = test_model(minibatch_index)
    		# print ('loss %f\n'%minibatch_avg_cost)
def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    sgd_optimization_mnist()