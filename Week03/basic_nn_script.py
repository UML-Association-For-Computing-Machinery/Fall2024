#!/usr/bin/python3

import numpy as np
import cv2 as cv2
import random

from skimage import color
from skimage import io
import matplotlib.pyplot as plt

#inputs and outputs
img0 = io.imread("cat.jpg", 0) # get image
img1 = io.imread("cat2.jpg", 0) # get image
img2 = io.imread("dog.jpg", 0) # get image
img3 = io.imread("cat3.jpg", 0) # get image
img4 = io.imread("dog2.jpg", 0) # get image
img5 = io.imread("cat4.jpg", 0) # get image
img6 = io.imread("bird.jpg", 0) # get image



ans = [1, 1, 0, 1, 0, 1, 0] # output array
x = [img0, img1, img2, img3, img4, img5, img6]

y = []
for i in range(len(x)):
    if ans[i]==1:
        y.append(np.ones(img0.shape).flatten().T)
    else:
        y.append(np.zeros(img0.shape).flatten().T)


for i in range(0, len(y)):
    #x[i] = x[i].reshape(-1,1)
    x[i] = np.resize(x[i],x[0].shape)
    #x[i] = x[i].flatten().T # convert 2d array into 1d array, and take transpose to get column vector
    #x[i] = np.linalg.norm(x[i], axis=1, keepdims=True) # 2d normalized array for image
print("image size: ", (x[0].shape))
print("training examples: ", np.array(x).shape)
x = np.array(x)
# objective: decide if cat or not
# neural network layer 1 - base features: color, size, shape  (3 nodes)
# neural network layer 2 - detailed features: 
#      color --> eye, mouth, body, feet
#      size --> whisker, eye, body, mouth, feet
#      shape --> eye, body, mouth, feet
#      therefore we have 4 nodes
# neural network layer 3 - aspects of each feature: hair size, hair pattern, body outline, mouth shape (4 nodes)
# neural network layer 4 - possible smaller features: tooth shape, whiskers, expression (3 nodes)
# neural network layer 5 - output layer: 1 node
# z[l] = output of activation from layer l
# n[1] = 4, n[2] = 4, n[3] = 3, n[4] = 1, n[0] = 3 input features
# z[l] = 3d vector (n[l] x 1)


m = (x.shape[0]) # number of training samples
learning_rate = 0.05 # learning rate
print("number of training examples: ", m)

l = 5 # number of layers in the network
n = np.zeros((5,1)) # layer dimensions for network

#normalizing inputs
for i in range (len(y)):
    x[i] = x[i]/255
    print("size is ", x[i].shape)
    
x_flatten = []
for i in range((x.shape[0])):
    x_flatten.append(x[i][:,:,0].flatten())
    
print ("training x's shape: " + str(x_flatten[0].shape))
n[0] = x_flatten[0].shape[0]
n[1] = 4
n[2] = 4
n[3] = 3
n[4] = 1

num_iterations = 100 # number of iterations for network
  
  
# INITIALIZING PARAMETERS
#biases to steer the entropy of the neural network, usually start at zero
b = []
b.append(np.zeros((int(n[0]), 1)))

#weight function, randomized in a normal distribution
w = []

costs = [] # costs per each iteration of the learning model

for i in range(1, l):
    b.append(np.zeros( (int(n[i]), 1 )))
    w.append(np.random.randn(int(n[i]), int(n[i-1])) * 0.01) 
    print("w",i, ": ", np.array(w[i-1]).shape)

for N in range(0, num_iterations):

    ##** Forward Propagation **##
    # logistic regression function: prediction output = function(weight transpose * input + bias)
    z = []

    # activation functions: 
    #    sigmoid = 1/1+e^(-z)
    #    relu = tanh(z)
    a = [] # np.zeros((l, len(y)))
    linear_cache = [] # np.zeros((l, len(y)))
    activation_cache = [] # np.zeros((l, len(y)))
    output_predictions = [] # np.zeros((l, len(y)))
    cache = [] # np.zeros((l, len(y)))
    
    cost = 0
    #for j in range(0, len(y)):
    activation_temp = []
    output_predictions_temp = []
    z_temp = []
    linear_cache_temp = []
    cache_temp = []
    
    a.append(np.array(x_flatten).T)
    #z.append(w)
    #output_predictions.append(w)
    print("x.shape = ", np.array(x_flatten).T.shape)
    #linear_cache.append(0)
    #cache.append([0,0])
    for j in range(1,l):
        print(np.array(w[j-1]).shape)
        print(np.array(b[j]).shape)
        print(np.array(a[j-1]).shape)
        print("j = ",j)
        print("weights", j," shape: ",w[j-1].shape)
        z.append( np.dot(w[j-1],a[j-1]) + b[j] )
        print("z[",j,"].shape = ", z[j-1].shape) 
        linear_cache.append([a[j-1], w[j-1], b[j]])
        a.append( 1/(1 + np.exp(-z[j-1])) )
        cache.append((linear_cache[j-1], a[j-1]))
        output_predictions.append(z[j-1])
        print("activations shape: " , a[j-1].shape)
        print("outputs shape: " , z[j-1].shape)
    #output_predictions.append(output_predictions_temp)
    print("output predictions shape: " ,len(output_predictions))
    print("linear cache shape: " , len(linear_cache))
    print("activation shape: " , len(a))
    print("z shape: " , len(z))

    #linear_cache.append(linear_cache_temp)
    #cache.append(cache_temp)

    # Measure cost: loss function to measure performance
    print("correct answers: ", ans)
    ans = np.array(ans)
    cost -= 1/ (x_flatten[0].shape[0]) * np.sum(np.multiply(ans, np.log(output_predictions[l-2])) + np.multiply((1 - ans), np.log(1 - output_predictions[l-2])))
    cost = np.squeeze(cost)     
    print("cost: ", cost)
    L = len(cache) # the number of layers
    m = len(output_predictions[l-2])
    
    #for a in range(len(ans)):
    #    if (ans[a] == 1):
    #        y[a] = np.ones((output_predictions[l-2].shape[0], output_predictions[l-2].shape[1]))
    #    else:
    #        y[a] = np.zeros((output_predictions[l-2].shape[0], output_predictions[l-2].shape[1]))
    #ans[j] = np.array(ans[j]).reshape(output_predictions[j][l-1].shape) # after this line, Y is the same shape as AL
    
    
    #for j in range (0, len(y)):
    # Initializing the backpropagation
    da = []
    y = np.zeros((1, (len(ans))))
    for i in range(int(len(ans))):
        y[0][i] = ans[i]
    print("ans shape: ", (1-y).shape)
    print("output predictions[l-2]: ", (1-output_predictions[l-2]).shape)
    da.append(- (np.divide(y, output_predictions[l-2]) - np.divide(1 - y, 1 - output_predictions[l-2]))) # derivative of cost with respect to AL

    # change in output of the function
    dz = []

    # change in weights for the function nodes
    dw = []

    # change in biases
    db = []
    # LINEAR -> SIGMOID BACKWARD (for L layers)
    # sigmoid backward
    #sigmoid_derivative = a[l-1] * (1 - a[l-1])
    
    #dz.append(np.matmul(da[0], sigmoid_derivative))
    # Propagate the error backward
    #  linear backward 
    
    #dw.append(1/m * np.matmul(dz, curr_cache.T))
    #db.append(1/m * np.sum(dz, keepdims = True, axis = 1))
    #da.append(np.matmul(da[0], dz)) # use for the next layer

    # LINEAR -> SIGMOID BACKWARD (for layer L-2 to 0)
    for i in range(1, l):
        #dz[i].append(a[i] - y)
        #for j in range(int(n[0])):
        #    dw[j] += (x[j][i]*dz[i])
        #db[i] = 1/m * np.sum(dz[i], axis = 1, keepdims = True)
        #dw[i] = 1/m * np.matmul(dz[i],np.transpose(a[i-1]))
        
        # sigmoid backward
        print("da shape: ", da[i-1].shape)
        sigmoid_derivative = np.matmul(a[l-i].T, (1 - a[l-i]))
        print("sigmoid derivative shape: ", sigmoid_derivative.shape)
        dz.append(np.dot(da[i-1], sigmoid_derivative))
        print("dz size: ", len(dz[0]))
        
        #  linear backward 
        m = len(cache[i-1])
        dw.append(1/m * np.matmul(dz[i-1], a[l-i-1].T))
        db.append(1/m * np.sum(dz[i-1], keepdims = True, axis = 1))
        print("w shape: ", w[l-i-1].T.shape)
        print("dz shape: ", np.array([dz[i-1]]).shape)
        da.append(np.matmul(w[l-i-1].T, dz[i-1])) # use for the next layer
        
        print("change in output (dz): " , dz[i-1])
        print("change in bias (db): " , db[i-1])
        print("change in weights (dw): " , dw[i-1])
        print("change in activation (da): " , da[i])
        
        print("output shape (dz): " , z[l-i-1].shape)
        print("bias shape (db): " , b[l-i].shape)
        print("weights shape (dw): " , w[l-i-1].shape)
        print("activation shape (da): " , a[l-i].shape)
        print("change in output shape (dz): " , dz[i-1].shape)
        print("change in bias shape (db): " , db[i-1].shape)
        print("change in weights shape (dw): " , dw[i-1].shape)
        print("change in activation shape (da): " , da[i-1].shape)
        
    for i in range(0, l-1):
        print("weights shape (dw): " , w[i].shape)
        print("change in weights shape (dw): " , dw[i].shape)
    # updating the parameters of the network
    for i in range(1, l):
        a[l-i] = a[l-i] - learning_rate*da[i-1]
        w[l-i-1] = w[l-i-1] - learning_rate*dw[i-1]
        b[l-i] = b[l-i] - learning_rate*db[i-1]
        
    costs.append(cost)
iterations = []
for N in range(num_iterations):
    iterations.append(N)
#plt.plot(iterations, costs)
#plt.xlabel('iteration number')
#plt.ylabel('costs of each iteration of network')
#plt.title('Costs of learning with learning rate = '+ str(learning_rate))
#plt.show()
    
    
## making predictions and evaluation display:
img = io.imread("esp32.jpg", 0)
img = np.resize(img,x[0].shape)
img = np.array(img)/255
img = img[:,:,0].flatten()
temp_z = []
temp_a = []
temp_output_predictions = []
temp_a.append(img)
for j in range(1,l):
    temp_z.append( np.dot(w[j-1],temp_a[j-1]) + b[j] )
    temp_a.append( 1/(1 + np.exp(-z[j-1])) )
    temp_output_predictions.append(z[j-1])
    print("activations shape: " , temp_a[j-1].shape)
    print("outputs shape: " , temp_z[j-1].shape)
output_sample = temp_output_predictions[l-2]
probability = (temp_a[l-1])
print("output is ", output_sample)
print("probability of being a cat is ", probability)
