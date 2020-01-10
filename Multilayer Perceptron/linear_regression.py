"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    N=X.shape[0]
    err=(1/N)*np.sum(np.abs(X@w-y))
#    err = None
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################	 	
  w = np.linalg.inv(X.T@X)@X.T@y 
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
#    w = np.zeros((X.shape[1]))
    L=X.T@X
    e,v=np.linalg.eig(L)
    if abs(np.min(e))<pow(10,-5):  
        while abs(np.min(e))<0.00001:
            L=X.T@X+0.1*np.eye(X.shape[1])
            e,v=np.linalg.eig(L)  
    w=np.linalg.inv(L)@X.T@y
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
      #####################################################
      # TODO 4: Fill in your code here #
      #####################################################		
    assert lambd!=None
    w = np.zeros((X.shape[1]))
#    print(lambd)
    w=np.linalg.inv(X.T@X+lambd*np.eye(X.shape[1]))@X.T@y
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    maxval=99999.99
#    cnt=0
    for i in range(-19,20):
        w=regularized_linear_regression(Xtrain, ytrain, pow(10,i))
        err=mean_absolute_error(w, Xval, yval)      
        if err<maxval:
            maxval=err
            bestlambda=pow(10,i)
#            cnt+=1
#    print('count='+str(cnt))
    #####################################################		
    
#    bestlambda = None
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    X1=X
    for i in range(2,power+1):
        temp=np.power(X,i)
        X1=np.hstack((X1,temp))      
    return X1


