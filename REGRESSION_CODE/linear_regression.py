import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    #err = None
    err = np.mean((np.dot(X, np.transpose(w)) - y) ** 2)
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		
  covariance_matrix = np.dot(X.transpose(), X)
  covariance_matrix_inverse = np.linalg.inv(covariance_matrix)
  w = np.dot(covariance_matrix_inverse, np.dot(X.transpose(), y))
  
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    covariance_matrix = np.dot(X.transpose(), X)
    lambda_added_covaMatrix = covariance_matrix + \
        np.identity(covariance_matrix.shape[0]) * lambd
    covariance_matrix_inverse = np.linalg.inv(lambda_added_covaMatrix)
    w = np.dot(covariance_matrix_inverse, np.dot(X.transpose(), y))
    
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    min_err = np.inf
    bestlambda = -1
    for i in range(14, -1, -1):
        lamda = 2 ** (-i)
        w = regularized_linear_regression(Xtrain, ytrain, lamda)
        err = mean_square_error(w, Xval, yval)
        #print("MSE is : ", err)
        if (err < min_err):
            min_err = err
            bestlambda = lamda

    return bestlambda
 
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    
    dim = X.shape[1]
    K = X
    for i in range(2, p+1):
        L = np.power(X, i)
        # print(L)
        #K = np.insert(K, K.shape[1], L, axis=1)
        K = np.concatenate((K, L), axis=1)
        # print(K)

    return K
   

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

