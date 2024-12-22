#import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d, correlate2d
import cv2 as cv2
from skimage.exposure import rescale_intensity
import scipy
from sklearn.preprocessing import normalize
import os

class Convolution:
    def __init__(self, image, kernel_size, num_filters):
        # Take in an image:
        self.image = (image[:,:,0])
        self.num_filters = num_filters
        img_width, img_height = image[:,:,0].shape
        self.kernel_shape = (num_filters, (kernel_size), (kernel_size)) 
        self.output_shape = (num_filters, img_height + kernel_size + 1, img_width + kernel_size + 1)
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(int(self.kernel_shape[1]), int(self.kernel_shape[2]))
        self.biases = np.random.randn(self.output_shape[1], self.output_shape[2])
  
        K = 100 # number of kernels


    # def corr2d(self, image, kernel):
        # output = np.zeros((image.shape), dtype="float32")
        # image_padded = np.zeros((image.shape[0] + int(self.kernel_size), image.shape[1] + int(self.kernel_size)))
        # image_padded[0:-self.kernel_size, 0:-self.kernel_size] = image

        # # Convolutional layer
        # for x in range(image.shape[0]): # for all of the width of the image
            # for y in range(image.shape[1]): # for all of the height of the image
                # #print(kernel)
                # if (image_padded[y : y + self.kernel_size, x : x + self.kernel_size].shape==output[y : y + self.kernel_size, x : x + self.kernel_size].shape and image_padded[y : y + self.kernel_size, x : x + self.kernel_size].shape[0]>0 and y + self.kernel_size < image.shape[1]-1 and x + self.kernel_size < image.shape[0]-1):
                    # #print(y,  y + self.kernel_size, image.shape[1])
                    # #print(x , x + self.kernel_size, image.shape[0])
                    # #print(output[y : y+self.kernel_size, x : x+self.kernel_size].shape)
                    # #print(image_padded[y : y + self.kernel_size, x : x + self.kernel_size].shape)
                    # k = (kernel * image_padded[y : y + self.kernel_size, x : x + self.kernel_size]) / (self.kernel_size * self.kernel_size)
                    # output[y : y+self.kernel_size, x : x+self.kernel_size] = k # convoluted value in top left of image, output (x, y) value
                # #output[y, x] = (kernel * image_padded[y: y+self.kernel_size, x: x+self.kernel_size]).sum()
        # # rescale the output image to be in the range [0, 255]
        # output = rescale_intensity(output, in_range=(0, 255))
        # output = (output / 255).astype("uint8")
        # return output
        
    def corr2d(self, image, kernel):
        # Flip the kernel
        kernel = np.flipud(np.fliplr(kernel))
        # convolution output
        output = np.zeros_like(image)

        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0] + len(kernel), image.shape[1] + len(kernel)))
        image_padded[0:-len(kernel), 0:-len(kernel)] = image

        # Loop over every pixel of the image
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                # element-wise multiplication of the kernel and the image
                output[y, x] = (kernel * image_padded[y: y+len(kernel), x: x+len(kernel)]).sum()

        return output
        
        
    def forward(self, input_data): # Implement both convolution of the input image and max pooling to remove noise
        self.input_data = input_data
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            print(self.input_data[:,:,0].shape)
            print(self.kernels)
            output = correlate2d(self.input_data[:,:,0], self.kernels)
        output = np.maximum(output, 0)
        return output
  
    def backward(self, dLayer_dOutput, learning_rate): # passes the gradients (partial derivatives) with respect to the loss function backward through the convolution layer.
    # we need to find the gradient of loss with respect to weights or kernels and the gradient of loss with respect to inputs
        dLayer_dInput = np.zeros_like(self.input_data)
        dLayer_dKernel = np.zeros_like(self.kernels)
        for i in range(self.num_filters):
            dLayer_dKernel[i] = correlate2d(self.input_data, dLayer_dOutput[i])
            dLayer_dInput +=  correlate2d( dL_dOutput, self.kernels[i] )
        self.kernels = self.kernels - mu * dLayer_dInput
        self.biases = self.biases - mu * dLayer_dBias
        return dLayer_dInput
        
        
class Pool:
    def __init__ (self, pool_size):
        self.pool_size = pool_size
    
    def forward(self, input_data):
        print(input_data.shape)
        self.input_data = input_data
        self.input_height, self.input_width = input_data.shape
        self.output_width = self.input_width / self.pool_size # change the output image to be in terms of the pooling results
        self.output_height = self.input_height / self.pool_size # change the output image to be in terms of the pooling results
        self.num_channels = 1
        self.output = np.zeros((int(self.output_width), int(self.output_height))) #, self.num_channels))
        #for c in range(self.num_channels):
        for x in range(int(self.output_width)):
            for y in range(int(self.output_height)):
                pool = self.input_data[y:y+self.pool_size, x:x+self.pool_size] #pool = self.input_data[c, y:y+self.pool_size, x:x+self.pool_size]
                self.output[x, y] = np.max(pool)#self.output[x, y, c] = np.max(pool)
        return self.output            
        
    def backward(self, dLayer_dOutput, learning_rate):
        dLayer_dInput = np.zeros_like(self.input_data)
        
        #for c in range(self.num_channels):
        for x in range(int(self.output_width)):
            for y in range(int(self.output_height)):
                pool = self.input_data[x:x+self.pool_size, y:y+self.pool_size]#, c] # form a pooled map
                mask = (pool == np.max(pool)) # check where the maximum value of the pooled map is to see which values to send to the next layer (equal to max) and which to discard (not equal to max)             
                dLayer_dInput[x:x+self.pool_size, y:y+self.pool_size] = dLayer_dOutput[x, y] * mask # confirm that the output matches the input and apply the mask on the change in the input                    
                #dLayer_dInput[x:x+self.pool_size, y:y+self.pool_size, c] = dLayer_dOutput[x, y, c] * mask # confirm that the output matches the input and apply the mask on the change in the input
        return dLayer_dInput        
        
class Fully_Connected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(self.output_size, self.input_size)
        self.biases = np.random.rand(self.output_size, 1)
   
    
    def softmax(self, output):
        # collapses the list of outputs in the fully connected map to a list of probabilites for each type of output from the neural network 
        # Shift the input values to avoid numerical instability
        shifted_output = output - np.max(output)
        exp_values = np.exp(shifted_output)
        sum_exp_values = np.sum(np.exp(abs(shifted_output)))
        log_sum_exp = np.log(sum_exp_values)

        # Compute the softmax probabilities
        probabilities = normalize(exp_values)
        print("softmax prob: ", probabilities)
        return probabilities
        
    def dSoftmax(self, softmax_func):  # in backpropagation, we need to get the gradients of loss with respect to the output to see if the network is learning
         # flatten the softmax function as a diagonal to isolate the loss parameters in the softmax output for probabilities of outputs
         # we are trying to sum all probabilities of each output in the model to 1
        print(np.diagflat(softmax_func) - np.dot(softmax_func, softmax_func.T))
        return np.diagflat(softmax_func) - np.dot(softmax_func, softmax_func.T)#np.diagflat(softmax_func) - np.dot(softmax_func, softmax_func.T)
        
    def forward(self, input_data):
        # output = z = w.T * x + b
        self.input_data = input_data
        print(input_data.shape)
        print(self.weights.shape)
        input_data_1D = self.input_data.flatten().reshape(-1,1)
        self.z = np.dot(self.weights, self.input_data.T) + self.biases
        self.output = self.softmax(self.z)
        print("forward softmax: ", self.output)
        return self.output
        
    def backward(self, dLayer_dOutput, learning_rate):
        # derivative of loss gradient with respect to pre-activation layer output
        print("soft max output: ", scipy.special.softmax(self.output))
        dLoss_dZ = np.dot(self.dSoftmax(self.output).flatten(), dLayer_dOutput)
        # derivative of loss with respect to the weights, dependent on the flattened input and change in loss with respect to the output
        print(dLoss_dZ)
        print("input data shape " , self.input_data.shape)
        dLoss_dWeight = np.dot(dLoss_dZ, self.input_data.flatten().reshape(-1,1))
        # derivative of the loss with respect to biases
        dLoss_dBiases = dLoss_dZ
        # derivative of loss with respect to input data, and reshape to input data shape
        dLoss_dInput = np.dot(self.weights.T, dLoss_dZ).reshape(self.input_data.shape) 
        # update weights and biases based on learning rates
        self.weights = self.weights - learning_rate * dLoss_dWeight
        self.biases = self.biases - learning_rate * dLoss_dBiases
        
        
    def cross_example_loss(self, predictions, targets, num_samples): # correlate between examples of different types and find the error
        cost = -1/num_samples * np.sum(np.multiply(targets, np.log(predictions)) + np.multiply((1 - targets), np.log(1 - predictions)))
        cost = np.squeeze(cost)   
        return cost
        
    def cross_example_loss_gradient(self, actual_labels, predicted_probs, num_samples):
        #num_samples = actual_labels.shape[0]
        gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples
        return gradient



def train_network(X, y, conv, pool, full, lr=0.01, epochs=200):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for i in range(len(X)):
            # Forward pass
            conv_out = conv.forward(X[i]) # forward propagation of network with new image
            print("conv output: ", conv_out)
            pool_out = pool.forward(conv_out) # pool the new image into maps 
            print("pool output: ", pool_out)
            full_out = full.forward(pool_out) # form fully connected maps of predictions with pooled images
            loss = full.cross_example_loss(full_out.flatten(), y[i], epochs) # find loss between the predictions and actual outputs
            total_loss += loss

            # Converting to One-Hot encoding
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])

            if num_pred == num_y:
                correct_predictions += 1
            # Backward pass
            gradient = full.cross_example_loss_gradient(y[i], full_out.flatten(), num_pred).reshape((-1, 1))
            full_back = full.backward(gradient, lr)
            pool_back = pool.backward(full_back, lr)
            conv_back = conv.backward(pool_back, lr)

        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")
        
def predict(input_sample, conv, pool, full):
    # Forward pass through Convolution and pooling
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    # Flattening
    flattened_output = pool_out.flatten()
    # Forward pass through fully connected layer
    predictions = full.forward(flattened_output)
    return predictions
    
    
#inputs and outputs
X_train = []
ans = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # output array
path = "test_imgs_cat"
files = os.listdir(path)
for item in files:
	X_train.append(imread(os.path.basename(path)+'/'+item))
#X_train = [img0, img1, img2, img3, img4, img5, img6]
for i in range(len(X_train)):
	X_train[i] = np.resize(X_train[i], X_train[0].shape)
conv = Convolution(X_train[0], 6, 1)
pool = Pool(2)
full = Fully_Connected(513,770)#(121, 10)

train_network(X_train, ans, conv, pool, full)