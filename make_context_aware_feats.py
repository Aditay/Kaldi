import numpy
import scipy.io as sio

data = sio.loadmat('all_data.mat')
data = data['all_data']

labels = numpy.load('labels.npy')

print(labels.shape)
print(data.shape)

contxt_labels = numpy.memmap('context_labels.dat', dtype='float32', mode='w+', shape=(1128094, 31*160))
print(contxt_labels[1, :])
a,b = data.shape
# context_lab = numpy.zeros((1,160*31))
context_lab = numpy.zeros((160))
context = [-15,-14,-13,12,-11,-10,-9,-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for i in xrange(a):
    if i == 0:
        for j in xrange(15):
            # context_lab[j*(160): (j+1)*160] = numpy.zeros((1,160))
            arr = numpy.zeros((160))
            context_lab = numpy.hstack((context_lab, arr))
        for j in range(16):
            arr = data[j]
            context_lab = numpy.hstack((context_lab, arr))
            # context_lab[(j+15)*160: (j+15+1)*160] = data[j]
            context_lab = context_lab[160:160*31 + 160]
        contxt_labels[i] = numpy.transpose(context_lab)
    elif i > (a - 10):
        context_lab = numpy.hstack((context_lab, numpy.zeros(1, 160)))
        contxt_labels[i] = numpy.transpose(context_lab[160:160*31 + 160])
    else:
        context_lab = numpy.hstack((context_lab, data[i+1]))
        contxt_labels[i] = numpy.transpose(context_lab[160:160*31+160])
    if i% 10000 == 0:
        print (i)


    # index = [i+j for j in context]
    # # print index
    # for j in index:
    #     if j < 0 or j >(a-1):
    #         arr = numpy.zeros(( 160))
    #         context_lab = numpy.hstack((context_lab, arr))
    #     else:
    #         arr = data[j]
    #         # print arr.shape
    #         # print context_lab.shape
    #         context_lab = numpy.hstack((context_lab, arr))
    # # print context_lab.shape
    # # print j
    # t = context_lab[160:2720+160]
    # contxt_labels[i] = numpy.transpose(t)
print ('done')
print(contxt_labels.shape)
print(contxt_labels[1,:])


    # if i == 0:
        # contxt_labels = numpy.hstack(())
