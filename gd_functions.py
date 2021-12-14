import numpy as np # imports a fast numerical programming library
import scipy as sp # imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm # allows us easy access to colormaps
import matplotlib.pyplot as plt # sets up plotting under plt
import pandas as pd # lets us handle data as dataframes
from sklearn.datasets import make_classification
import seaborn as sns


def sigmoid(x):
    '''
    Function to compute the sigmoid of a given input x.
    
    Input:
    x: it's the input data matrix. The shape is (N, H)
    Output:
    g: The sigmoid of the input x
    '''
    g = 1 / (1 + np.exp(-x))
    return g

def log_likelihood(theta,features,target):
    '''
    Function to compute the log likehood of theta according to data x and label y
    
    Input:
    theta: it's the model parameter matrix.
    features: it's the input data matrix. The shape is (N, H)
    target: the label array
    
    Output:
    log_g: the log likehood of theta according to data x and label y
    '''
      
    log_l=((target * np.log(sigmoid(features @ theta))+(1-target)*np.log(1-sigmoid(features @ theta))).sum()) / len(features)

    return log_l


def predictions(features, theta):
    '''
    Function to compute the predictions for the input features
    
    Input:
    theta: it's the model parameter matrix.
    features: it's the input data matrix. The shape is (N, H)
    
    Output:
    preds: the predictions of the input features
    '''
      
    response=sigmoid(features @ theta)
    preds = np.where(response >=0.5, 1, 0)
    return preds


def update_theta(theta, target, preds, features, lr):
    '''
    Function to compute the gradient of the log likelihood
    and then return the updated weights
    Input:
    theta: the model parameter matrix.
    target: the label array
    preds: the predictions of the input features
    features: it's the input data matrix. The shape is (N, H)
    lr: the learning rate
    
    Output:
    theta: the updated model parameter matrix.
    '''
    
    # (712, 1) @ (712, 8)
    log_lik_deriv = (((target - sigmoid(features @ theta)).transpose() @ features).transpose() / len(features))
    theta = theta + lr * log_lik_deriv
    return theta 

def gradient_ascent(theta, features, target, lr, num_steps):
    '''
    Function to execute the gradient ascent algorithm
    Input:
    theta: the model parameter matrix.
    target: the label array
    num_steps: the number of iterations 
    features: the input data matrix. The shape is (N, H)
    lr: the learning rate
    
    Output:
    theta: the final model parameter matrix.
    log_likelihood_history: the values of the log likelihood during the process
    '''

    log_likelihood_history = np.zeros(num_steps)
    
    
    for step in range(num_steps):
      
        log_likelihood_history[step]=log_likelihood(theta,features,target)
        preds=predictions(features, theta)
        theta=update_theta(theta, target, preds, features, lr)
    
    return theta, log_likelihood_history
