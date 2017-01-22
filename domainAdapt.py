import numpy
import theano
from theano import tensor
import cPickle
from io_func.data_io import read_dataset, read_data_args

import scipy.io


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
print test_set_y.shape

n_in = 440
n_hidden = 1024
n_out = 1940

x = tensor.matrix('x')
# W1_value = a[0]
# b1_value = a[1]
# W2_value = a[2]
# b2_value = a[3]
# W3_value = a[4]
# b3_value = a[5]
# W4_value = a[6]
# b4_value = a[7]
# W_out_value = a[8]
# b_out_value = a[9]
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


W1 = theano.shared(value=W1_value,
                   name='W1',
                   borrow=False)
b1 = theano.shared(value=b1_value,
                   name='b1',
                   borrow=False)
W2 = theano.shared(value=W2_value,
                   name='W2',
                   borrow=False)
b2 = theano.shared(value=b2_value,
                   name='b2',
                   borrow=False)
W3 = theano.shared(value=W3_value,
                   name='W3',
                   borrow=False)
b3 = theano.shared(value=b3_value,
                   name='b3',
                   borrow=False)
W4 = theano.shared(value=W4_value,
                   name='W4',
                   borrow=False)
b4 = theano.shared(value=b4_value,
                   name='b4',
                   borrow=False)
W_out = theano.shared(value=W_out_value,
                      name='W_out',
                      borrow=False)
b_out = theano.shared(value=b_out_value,
                      name='b_out',
                      borrow=False)
# l1 = theano.shared(value=l1_value,
#                    name='l1',
#                    borrow=True)
# l2 = theano.shared(value=l2_value,
#                    name='l2',
#                    borrow=True)
# l3 = theano.shared(value=l3_value,
#                    name='l3',
#                    borrow=True)
# l4 = theano.shared(value=l4_value,
#                    name='l4',
#                    borrow=True)

y = tensor.lvector('y')

hidden1 = tensor.nnet.sigmoid(tensor.dot(x, W1) + b1)
hidden2 = tensor.nnet.sigmoid(tensor.dot(hidden1, W2) + b2)
hidden3 = tensor.nnet.sigmoid(tensor.dot(hidden2, W3) + b3)
hidden4 = tensor.nnet.sigmoid(tensor.dot(hidden3, W4) + b4)
hidden_out = tensor.dot(hidden4, W_out) + b_out

p_y_given_x = tensor.nnet.softmax(hidden_out)
y_pred = tensor.argmax(p_y_given_x, axis=1)
log_prob = tensor.log(p_y_given_x)
log_likelihood = log_prob[tensor.arange(y.shape[0]), y]

# alpha1 = 0.01
# alpha2 = 0.01
# alpha3 = 0.01
# alpha4 = 0.01
# loss1 = - log_likelihood.mean() + alpha1*((hidden1 - tensor.dot(x, W1_value) - b1_value)**2).sum() + alpha2*((hidden2 - tensor.dot(hidden1, W2_value) - b2_value)**2).sum() + alpha3*((hidden3 - tensor.dot(hidden2, W3_value) - b3_value)**2).sum() + alpha4*((hidden4 - tensor.dot(hidden3, W4_value) - b4_value)**2).sum()
loss1 = -log_likelihood.mean()


misclass_nb = tensor.neq(y_pred, y)
misclass_rate = misclass_nb.mean()
test_model = theano.function(inputs=[x,y],
                             outputs=misclass_rate)
# y_predictor =
train_error = test_model(train_set_x, train_set_y)
test_error = test_model(test_set_x, test_set_y)

print ('Train error: %f'%(train_error*100))
print ('Test error: %f'%(test_error*100))
# LHUC network is defined here

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
        high=-numpy.sqrt(6./(n_hidden+n_hidden)),
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

hidden1_l = tensor.nnet.sigmoid(tensor.dot(x, W1) + b1)*2*tensor.nnet.sigmoid(l1)
hidden2_l = tensor.nnet.sigmoid(tensor.dot(hidden1_l, W2) + b2)*2*tensor.nnet.sigmoid(l2)
hidden3_l = tensor.nnet.sigmoid(tensor.dot(hidden2_l, W3) + b3)*2*tensor.nnet.sigmoid(l3)
hidden4_l = tensor.nnet.sigmoid(tensor.dot(hidden3_l, W4) + b4)*2*tensor.nnet.sigmoid(l4)
hidden_out_l = tensor.dot(hidden4_l, W_out) + b_out

p_y_given_x_lhuc = tensor.nnet.softmax(hidden_out_l)
y_pred_lhuc = tensor.argmax(p_y_given_x_lhuc, axis=1)
log_prob_lhuc = tensor.log(p_y_given_x_lhuc)
log_likelihood_lhuc = log_prob_lhuc[tensor.arange(y.shape[0]), y]
loss_lhuc = - log_likelihood_lhuc.mean()

misclass_nb_lhuc = tensor.neq(y_pred_lhuc, y)
misclass_rate_lhuc = misclass_nb_lhuc.mean()
test_model_lhuc = theano.function(inputs=[x, y],
                                  outputs=misclass_rate_lhuc)


n_in_a = 1024
n_hidden_a = 512
n_out_a = 2

Wa1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_in_a+n_hidden_a)),
        high=numpy.sqrt(6./(n_in_a+n_hidden_a)),
        size=(n_in_a, n_hidden_a)
    ),
    dtype=theano.config.floatX
)

ba1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden_a+n_in_a)),
        high=numpy.sqrt(6./(n_in_a+n_hidden_a)),
        size=(n_hidden_a,)
    ),
    dtype=theano.config.floatX
)

Wa2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden_a+n_hidden_a)),
        high=numpy.sqrt(6./(n_hidden_a+n_hidden_a)),
        size=(n_hidden_a, n_hidden_a)
    ),
    dtype=theano.config.floatX
)

ba2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden_a+n_hidden_a)),
        high=numpy.sqrt(6./(n_hidden_a+n_hidden_a)),
        size=(n_hidden_a,)
    ),
    dtype=theano.config.floatX
)

Wa_out_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden_a+n_out_a)),
        high=numpy.sqrt(6./(n_hidden_a+n_out_a)),
        size=(n_hidden_a, n_out_a)
    ),
    dtype=theano.config.floatX
)

ba_out_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(n_hidden_a+n_out_a)),
        high=numpy.sqrt(6./(n_hidden_a+n_out_a)),
        size=(n_out_a,)
    ),
    dtype=theano.config.floatX
)

Wa1 = theano.shared(value=Wa1_value,
                    name='Wa1',
                    borrow=True)
ba1 = theano.shared(value=ba1_value,
                    name='ba1',
                    borrow=True)
Wa2 = theano.shared(value=Wa2_value,
                    name='Wa2',
                    borrow=True)
ba2 = theano.shared(value=ba2_value,
                    name='ba2',
                    borrow=True)
Wa_out = theano.shared(value=Wa_out_value,
                       name='Wa_out',
                       borrow=True)
ba_out = theano.shared(value=ba_out_value,
                       name='ba_out',
                       borrow=True)

y2 = tensor.lvector('y2')
# get_flag = theano.function(inputs=[y2],
#                            outputs=[y2])
# flag = get_flag(y2)
# flag = y2.eval()
# flag = y2.get_scalar_constant_value()
# fl = tensor.lvector('flag')
# if flag == 1:
hiddena_1 = tensor.nnet.sigmoid(tensor.dot(hidden4_l, Wa1) + ba1)
# else:
#     hiddena_1 = tensor.nnet.sigmoid(tensor.dot(hidden4, Wa1) + ba1)

hiddena_2 = tensor.nnet.sigmoid(tensor.dot(hiddena_1, Wa2) + ba2)
hiddena_out = tensor.dot(hiddena_2, Wa_out) + ba_out
p_y_given_x_adapt = tensor.nnet.softmax(hiddena_out)
y_pred_adapt = tensor.argmax(p_y_given_x_adapt, axis=1)
log_prob_adapt = tensor.log(p_y_given_x_adapt)
log_likelihood_adapt = log_prob_adapt[tensor.arange(y2.shape[0]), y2]
loss_adapt = - log_likelihood_adapt.mean()


misclass_nb_adapt = tensor.neq(y_pred_lhuc, y2)
misclass_rate_adapt = misclass_nb_adapt.mean()
test_model_adapt = theano.function(inputs=[x, y2],
                                   outputs=misclass_rate_adapt)
alpha = 0.1
g_l1 = tensor.grad(cost=(loss_lhuc - alpha*loss_adapt), wrt=l1)
g_l2 = tensor.grad(cost=(loss_lhuc - alpha*loss_adapt), wrt=l2)
g_l3 = tensor.grad(cost=(loss_lhuc - alpha*loss_adapt), wrt=l3)
g_l4 = tensor.grad(cost=(loss_lhuc - alpha*loss_adapt), wrt=l4)
# g_lout = tensor.grad(cost=loss_lhuc, wrt=)
g_Wa1 = tensor.grad(cost=loss_adapt, wrt=Wa1)
g_Wa2 = tensor.grad(cost=loss_adapt, wrt=Wa2)
g_Waout = tensor.grad(cost=loss_adapt, wrt=Wa_out)
g_ba1 = tensor.grad(cost=loss_adapt, wrt=ba1)
g_ba2 = tensor.grad(cost=loss_adapt, wrt=ba2)
g_baout = tensor.grad(cost=loss_adapt, wrt=ba_out)

learning_rate = numpy.float32(0.13)
new_l1 = l1 - learning_rate*g_l1*tensor.transpose(y2)
new_l2 = l2 - learning_rate*g_l2*tensor.transpose(y2)
new_l3 = l3 - learning_rate*g_l3*tensor.transpose(y2)
new_l4 = l4 - learning_rate*g_l4*tensor.transpose(y2)

new_Wa1 = Wa1 - learning_rate*g_Wa1
new_Wa2 = Wa2 - learning_rate*g_Wa2
new_Waout = Wa_out - learning_rate*g_Waout
new_ba1 = ba1 - learning_rate*g_ba1
new_ba2 = ba2 - learning_rate*g_ba2
new_baout = ba_out - learning_rate*g_baout

# if flag == 1:
update = [(l1, new_l1), (l2, new_l2), (l3, new_l3), (l4, new_l4), (Wa1, new_Wa1), (Wa2, new_Wa2), (Wa_out, new_Waout), (ba1, new_ba1), (ba2, new_ba2), (ba_out, new_baout)]
# else:
#     update = [(Wa1, new_Wa1), (Wa2, new_Wa2),
#               (Wa_out, new_Waout), (ba1, new_ba1), (ba2, new_ba2), (ba_out, new_baout)]

adverserial_adaptation = theano.function(inputs=[x, y, y2],
                                         outputs=loss_adapt,
                                         updates=update)
batch_size = 125
n_test_batches = test_set_x.shape[0] // batch_size
n_train_batches = train_set_x.shape[0] // batch_size
n_dev_batches = dev_set_x.shape[0] // batch_size

labels1 = numpy.zeros(4000)
l = test_set_x.shape[0]
labels2 = numpy.ones(l)
total_x = numpy.concatenate((train_set_x[0:4000], test_set_x), axis=0)
total_y = numpy.concatenate((train_set_y[0:4000], dev_set_y), axis=0)
total_y_original = numpy.concatenate((train_set_y[0:4000], test_set_y), axis=0)
total_y_domain = numpy.concatenate((labels1, labels2), axis=0)

numpy.random.seed(0)
print total_x.shape
a = numpy.arange(6790)

a = numpy.random.permutation(a)

total_x = total_x[a]
total_y = total_y[a]
total_y_domain = total_y_domain[a]
total_y_original = total_y_original[a]
total_y = total_y.astype(numpy.int64)
total_y_domain = total_y_domain.astype(numpy.int64)
total_y_original = total_y_original.astype(numpy.int64)
# trainx = total_x[0:4000]
# testx = total_x[4000:]
# trainy = total_y[0:4000]
# testy = total_y[4000:]
def get_minibatch(i, dataset_x, dataset_y, domain_y):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    batch_x =  dataset_x[start_idx:end_idx]
    batch_y = dataset_y[start_idx:end_idx]
    batch_domain = domain_y[start_idx:end_idx]
    return batch_x, batch_y, batch_domain

n_epoch = 50

epoch = 0

while epoch < n_epoch:
    epoch = epoch + 1
    for minibatch_index in range(10):
        minibatch_x, minibatch_y, batch_domain = get_minibatch(minibatch_index, total_x, total_y, total_y_domain)
        # flag = get_flag(batch_domain)
        minibatch_loss = adverserial_adaptation(minibatch_x, minibatch_y, batch_domain)
    lhuc_error = test_model_lhuc(total_x, total_y_original)
    domain_error = test_model_adapt(total_x, total_y_domain)
    print ('lhuc_error %f , domain error %f '%(lhuc_error*100, domain_error*100))


"""
na_in = 1024
na_hidden = 512
na_out = 2
Wa_1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(na_in+na_hidden)),
        high=numpy.sqrt(6./(na_in+na_hidden)),
        size=(na_in, na_hidden)
    ),
    dtype=theano.config.floatX
)

ba_1_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(na_in+na_hidden)),
        high=numpy.sqrt(6./(na_in+na_hidden)),
        size=(na_hidden,)
    ),
    dtype=theano.config.floatX
)

Wa_2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(na_hidden+na_hidden)),
        high= numpy.sqrt(6./(na_hidden+na_hidden)),
        size=(na_hidden, na_hidden)
    ),
    dtype=theano.config.floatX
)

ba_2_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(na_hidden+na_hidden)),
        high=numpy.sqrt(6./(na_hidden+na_hidden)),
        size=(na_hidden,)
    ),
    dtype=theano.config.floatX
)

Wa_out_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(na_hidden+na_out)),
        high=numpy.sqrt(6./(na_hidden+na_out)),
        size=(na_hidden, na_out)
    ),
    dtype=theano.config.floatX
)

ba_out_value = numpy.asarray(
    numpy.random.uniform(
        low=-numpy.sqrt(6./(na_hidden+na_out)),
        high=numpy.sqrt(6./(na_hidden+na_out)),
        size=(na_out,)
    ),
    dtype=theano.config.floatX
)

y2 = tensor.lvector('y2')

Wa_1 = theano.shared(value=Wa_1_value,
                     name='Wa_1',
                     borrow=True)
ba_1 = theano.shared(value=ba_1_value,
                     name='ba_1',
                     borrow=True)
Wa_2 = theano.shared(value=Wa_2_value,
                     name='Wa_2',
                     borrow=True)
ba_2 = theano.shared(value=ba_2_value,
                     name='ba_2',
                     borrow=True)
Wa_out = theano.shared(value=Wa_out_value,
                       name='ba_out',
                       borrow=True)

ba_out = theano.shared(value=ba_out_value,
                       name='ba_out',
                       borrow=True)

hidden_a1 = tensor.nnet.sigmoid(tensor.dot(hidden4, Wa_1) + ba_1)
hidden_a2 = tensor.nnet.sigmoid(tensor.dot(hidden_a1, Wa_2) + ba_2)
hidden_aout = tensor.dot(hidden_a2, Wa_out) + ba_out
p_y_given_x_adapted = tensor.nnet.softmax(hidden_aout)
y_pred_adapt = tensor.argmax(p_y_given_x_adapted, axis=1)
log_prob_adapt = tensor.log(p_y_given_x_adapted)
log_likelihood_adapted = log_prob_adapt[tensor.arange(y2.shape[0]), y2]

loss2 = -log_likelihood_adapted.mean()
"""
