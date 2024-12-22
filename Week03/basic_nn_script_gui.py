#!/usr/bin/python3

import numpy as np
import cv2 as cv2
import random
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
import os
import json

X_train = []
ans = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0] # output array 
cats = []
birds = []
sharks = []
other = []
path = "test_imgs_cat"
files = os.listdir(path)
dict = []
for item in files:
    X_train.append(io.imread(os.path.basename(path)+'/'+item))
    if 'tiger' in item or 'cat' in item:
        dict.append({item : 'cat'})
        cats.append(1)
        birds.append(0)
        sharks.append(0)
        other.append(0)
    elif ('bird') in item:
        dict.append({item : 'bird'})
        cats.append(0)
        birds.append(1)
        sharks.append(0)
        other.append(0)
    elif ('shark') in item:
        dict.append({item : 'shark'})
        cats.append(0)   
        birds.append(0)        
        sharks.append(1)
        other.append(0)
    else:
        dict.append({item : 'not an animal'})
        cats.append(0)   
        birds.append(0)  
        sharks.append(0)
        other.append(1)
labels = np.array([cats, birds, sharks, other], np.int32)
        
for i in range(0, len(X_train)):
    #x[i] = x[i].reshape(-1,1)
    X_train[i] = np.resize(X_train[i], X_train[0].shape)
    #x[i] = x[i].flatten().T # convert 2d array into 1d array, and take transpose to get column vector
    #x[i] = np.linalg.norm(x[i], axis=1, keepdims=True) # 2d normalized array for image
    
print("image size: ", (X_train[0].shape))
print("training examples: ", np.array(X_train).shape)
train_imgs = np.array(X_train)
print("x train length: ", len(X_train))
print("y length: ", labels.shape)

def Neural_network(x, ans):

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
    for i in range (len(x)):
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
    b = []
    w = []
    costs = [] # costs per each iteration of the learning model
    for j in range(ans.shape[0]):
    #biases to steer the entropy of the neural network, usually start at zero
        b_temp = []
        b_temp.append(np.zeros((int(n[0]), 1)))

        #weight function, randomized in a normal distribution
        w_temp = []
        for i in range(1, l):
            b_temp.append(np.zeros( (int(n[i]), 1 )))
            w_temp.append(np.random.randn(int(n[i]), int(n[i-1])) * 0.01) 
            print("w",i, ": ", np.array(w_temp[i-1]).shape)
        w.append(w_temp)
        b.append(b_temp)
    
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
        
        for i in range(0, ans.shape[0]):
            a_temp = []
            z_temp = []
            output_predictions_temp = []
            a_temp.append(np.array(x_flatten).T)       
            print("x.shape = ", np.array(x_flatten).T.shape)
            for j in range(1,l):
                print("j = ",j)
                print("weights", j," shape: ",w[i][j-1].shape)  
                print(np.array(w[i][j-1]).shape)
                print(np.array(b[i][j]).shape)
                print(np.array(a_temp[j-1]).shape)
                z_temp.append( np.dot(w[i][j-1], np.array(a_temp[j-1])) + np.array(b[i][j]) )
               
                print(z_temp)
                print("z[",j,"].shape = ", len(z_temp[j-1])) 
                linear_cache.append([a_temp[j-1], w[i][j-1], b[i][j]])
                a_temp.append( 1/(1 + np.exp(-z_temp[j-1])) )
                output_predictions_temp.append(z_temp[j-1])
                cache.append((linear_cache[j-1], a_temp[j-1]))            
                print("activations shape: " , np.array(a_temp[j-1]).shape)
                print("outputs shape: " , len(z_temp[j-1]))
            a.append((a_temp))
            z.append((z_temp))
            output_predictions.append((output_predictions_temp))
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
        #cost -= 1/ (x_flatten[0].shape[0]) * np.sum(np.multiply(ans, np.log(output_predictions[l-2])) + np.multiply((1 - ans), np.log(1 - output_predictions[l-2])))
        for i in range (len(output_predictions)):
            cost -= 1/ (x_flatten[0].shape[0]) * np.sum(np.multiply(ans, np.log(output_predictions[i][l-2])) + np.multiply((1 - ans), np.log(1 - output_predictions[i][l-2])))
        cost = np.squeeze(cost)     
        print("cost: ", cost)
        L = len(cache) # the number of layers
        m = len(output_predictions[0][l-2])
        
        #for a in range(len(ans)):
        #    if (ans[a] == 1):
        #        y[a] = np.ones((output_predictions[l-2].shape[0], output_predictions[l-2].shape[1]))
        #    else:
        #        y[a] = np.zeros((output_predictions[l-2].shape[0], output_predictions[l-2].shape[1]))
        #ans[j] = np.array(ans[j]).reshape(output_predictions[j][l-1].shape) # after this line, Y is the same shape as AL
        
        
        #for j in range (0, len(y)):
        # Initializing the backpropagation
        da = []
        y = np.zeros((int(ans.shape[0]), int(ans.shape[1])))
        y = ans
        #for i in range(int(ans.shape[0])):
        #    for j in range(int(ans.shape[1])):
        #        print(ans[0])
        #        print(ans[1])
        #        print(ans[2])
        #        print(ans[3])
        #        y[j][i] = ans[i][j]
        print("ans shape: ", (1-y).shape)
        for j in range(len(output_predictions)):
            print("output predictions[l-2]: ", (1-output_predictions[j][l-2]).shape)
            da.append(- (np.divide(y[j], output_predictions[j][l-2]) - np.divide(1 - y[j], 1 - output_predictions[j][l-2]))) # derivative of cost with respect to AL

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
                sigmoid_derivative = np.matmul(a[j][l-i].T, (1 - a[j][l-i]))
                print("sigmoid derivative shape: ", sigmoid_derivative.shape)
                dz.append(np.dot(da[i-1], sigmoid_derivative))
                print("dz size: ", len(dz[0]))
                
                #  linear backward 
                m = len(cache[i-1])
                dw.append(1/m * np.matmul(dz[i-1], a[j][l-i-1].T))
                db.append(1/m * np.sum(dz[i-1], keepdims = True, axis = 1))
                print("w shape: ", w[j][l-i-1].T.shape)
                print("dz shape: ", np.array(dz[i-1]).shape)
                da.append(np.matmul(w[j][l-i-1].T, dz[i-1])) # use for the next layer
                
                print("change in output (dz): " , dz[i-1])
                print("change in bias (db): " , db[i-1])
                print("change in weights (dw): " , dw[i-1])
                print("change in activation (da): " , da[i])
                
                print("output shape (dz): " , z[j][l-i-1].shape)
                print("bias shape (db): " , b[j][l-i].shape)
                print("weights shape (dw): " , w[j][l-i-1].shape)
                print("activation shape (da): " , a[j][l-i].shape)
                print("change in output shape (dz): " , dz[i-1].shape)
                print("change in bias shape (db): " , db[i-1].shape)
                print("change in weights shape (dw): " , dw[i-1].shape)
                print("change in activation shape (da): " , da[i-1].shape)
                
            for i in range(0, l-1):
                print("weights shape (dw): " , w[j][i].shape)
                print("change in weights shape (dw): " , dw[i].shape)
            # updating the parameters of the network
            for i in range(1, l):
                w[j][l-i-1] = w[j][l-i-1] - learning_rate*dw[i-1]
                b[j][l-i] = b[j][l-i] - learning_rate*db[i-1]
                
            costs.append(cost)
    iterations = []
    for N in range(num_iterations):
        iterations.append(N)
    #plt.plot(iterations, costs)
    #plt.xlabel('iteration number')
    #plt.ylabel('costs of each iteration of network')
    #plt.title('Costs of learning with learning rate = '+ str(learning_rate))
    #plt.show()
    with open('params.txt', 'w') as filehandle:
        json.dump(w.toList(), filehandle)
        json.dump(b.toList(), filehandle)
        json.dump(z.toList(), filehandle)
        json.dump(a.toList(), filehandle)
    return cost, m
    
    
## making predictions and evaluation display:
def predict(img, file, m):
    img = np.resize(img,m)
    img = np.array(img)/255
    img = img[:,:,0].flatten()
    temp_z = []
    temp_a = []
    temp_output_predictions = []
    w = []
    b = []
    a = []
    z = []
    temp_a.append(img)
    with open(file, 'r') as fd:
        reader = csv.reader(fd)
        w = np.array(reader[0])
        b = np.array(reader[1])
       
    for j in range(1,l):
        temp_z.append( np.dot(w[j-1],temp_a[j-1]) + b[j] )
        temp_a.append( 1/(1 + np.exp(-temp_z[j-1])) )
        temp_output_predictions.append(temp_z[j-1])
        print("activations shape: " , temp_a[j-1].shape)
        print("outputs shape: " , temp_z[j-1].shape)
    output_sample = temp_output_predictions[l-2]
    probability = (temp_a[l-1])
    print("output is ", output_sample)
    print("probability of being a cat is ", probability)

cost, m = Neural_network(train_imgs, labels)
predict(io.imread("esp32.jpg", 0), "params.txt", m)
