'''
Created on Dec 9, 2017

@author: go
'''

# from keras import backend as K

import cv2
import numpy as np

def gabor_init(shape, dtype=None):
    kernel_size_1,kernel_size_2,ch,num_filters = shape
    
    # guarentee kernel size to be odd
    kernel_size = -1
    if kernel_size_1 % 2 == 1:
        kernel_size = kernel_size_1
    elif kernel_size_2 % 2 == 1:
        kernel_size = kernel_size_2
    else:
        kernel_size = kernel_size_1 + 1
    
    # evaluate parameter interval based on number of convolution filters
    range_sigma = np.arange(5, ((kernel_size/2)+1), (((kernel_size/2)+1)-5)/(num_filters*1.))
    range_lambda = np.array([kernel_size-2] * num_filters)
    range_theta = np.arange(0, 360+1,(360+1)/(num_filters*1.))
    range_gamma = np.arange(100, 300+1,((300+1)-100)/(num_filters*1.))
    range_psi = np.arange(90, 360+1,((360+1)-90)/(num_filters*1.))
    
    kernels = []
    for i in range(num_filters):
        g_sigma = range_sigma[i]
        g_lambda = range_lambda[i] + 2
        g_theta = range_theta[i] * np.pi / 180.
        g_gamma = range_gamma[i] / 100.
        g_psi = (range_psi[i] - 180) * np.pi / 180
    
        print 'kern_size=' + str(kernel_size) + ', sigma=' + str(g_sigma) + ', theta=' + str(g_theta) + ', lambda=' + str(g_lambda) +', gamma=' + str(g_gamma) + ', psi=' + str(g_psi)
        kernel = cv2.getGaborKernel((kernel_size, kernel_size),
                                    g_sigma,
                                    g_theta,
                                    g_lambda,
                                    g_gamma,
                                    g_psi)
#         kernel_img = kernel/2.+0.5
#         cv2.imshow('Kernel', cv2.resize(kernel_img, (kernel_size*20, kernel_size*20)))
#         cv2.waitKey()
        kernels = kernels + kernel.ravel().tolist() * ch
        
    kernels = np.array(kernels)
        
    return kernels.reshape(shape)

# def gabor_init2(shape, dtype=None):
#     gain = 1.
#     xx = gain * np.identity(shape[0])
#     print '---'
#     print xx.shape
#     print type(xx)
#     return xx
# 
# def gabor_init1(shape, dtype=None):
#     gain = 1.
#     seed = None
#     
#     num_rows = 1
#     for dim in shape[:-1]:
#         num_rows *= dim
#     num_cols = shape[-1]
#     flat_shape = (num_rows, num_cols)
#     if seed is not None:
#         np.random.seed(seed)
#     a = np.random.normal(0.0, 1.0, flat_shape)
#     u, _, v = np.linalg.svd(a, full_matrices=False)
#     # Pick the one with the correct shape.
#     q = u if u.shape == flat_shape else v
#     q = q.reshape(shape)
#     xx = gain * q[:shape[0], :shape[1]]
# #     print '--------'
# #     _,_,_,n = xx.shape
# #     for i in range(n):
# #         print i
# #         print xx[:,:,:,i]
#     return xx

# def main():
#     shape = (11,11,1,96)
#     l = gabor_init(shape)
#     print(l)
#     print(l.shape)
#     print(type(l))
# 
# if __name__ == '__main__':
#     main()