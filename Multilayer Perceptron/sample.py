# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:00:40 2019

@author: ragha
"""
import numpy as np
class relu:

    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(X):

        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.
            
            Return:
            - forward_output: A numpy array of the same shape of X
        """
        
        ################################################################################
        # TODO: Implement the relu forward pass. Store the result in forward_output    #
        ################################################################################
#        self.mask=(X>0).astype(int)
        forward_output=np.maximum(X,0)
        raise NotImplementedError("Not Implemented function: forward, class: relu")
        return forward_output

    def backward(X, grad):

        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # You can use the mask created in the forward step.
        ####################################################################################################
#        grad[X>0]=1
        backward_output=grad*((X>0).astype(int))
        raise NotImplementedError("Not Implemented function: backward, class: relu")
        return backward_output


# 3. tanh Activation

class tanh:

        
    def forward(X):

        """
            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the tanh forward pass. Store the result in forward_output
        # You can use np.tanh()
        ################################################################################
#        forward_output=(2/(1+np.exp(-2*X))) - 1
        forward_output=np.tanh(X)
        raise NotImplementedError("Not Implemented function: forward, class: tanh")
        return forward_output

    def backward(self, X, grad):

        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """
        ####################################################################################################
        # TODO: Implement the backward pass
        # Derivative of tanh is (1 - tanh^2)
        ####################################################################################################
#        backward_output=1-pow(tanh.forward(X),2)
        backward_output=grad*(1-pow(np.tanh(X),2))
        raise NotImplementedError("Not Implemented function: backward, class: tanh")
        return backward_output

X=np.random.randint(0,10,(5,10))
t=tanh.forward(X)


