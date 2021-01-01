import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss                  # 
        ################################################
        def sign(x):
            if x >= 0:
                return 1
            else:
                return -1
            
        # 1. Convert 0-1 labels to -1 and 1 so that we can use our same loss function.
        y1= np.array(y)
        y1[y1 == 0] = -1
        
        for _ in range(max_iterations):
            z = np.dot(X, w) + b  # (N,)
            loss = y1 * z         # (N,)
            loss[loss <= 0] = -1
            loss[loss > 0] = 0
            error = loss * y1     # (N,)
            del_w = np.dot(error, X) / N  # (N,).(N,D) = (D,)
            assert del_w.shape == (D,)
            del_b = np.mean(error)
            w = w - step_size * del_w     # (D,)
            b = b - step_size * del_b
        
        

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for _ in range(max_iterations):
            z = np.dot(X, w) + b
            preds = sigmoid(z)
            error = preds - y
            w_del = np.dot(error, X) / N
            b_del = np.mean(error)
            w = w - step_size * w_del
            b = b - step_size * b_del
          

        

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    def sigmoid_helper(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    value = np.vectorize(sigmoid_helper)(z)
    
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    def sign_vec(x):
        convert = [0 for i in range(len(x))]
        for i in range(len(convert)):
            convert[i] = 1 if x[i] >= 0 else -1
        return convert

    preds = sign_vec(np.dot(X, w) + b)
    preds = np.asarray(preds)
    preds[preds == -1] = 0
  

    assert preds.shape == (N,) 
    return preds

def softmax1(X):

    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
      
    def softmax(x):
        e = np.exp(x - np.max(x))
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T

    """      
    X_mod = np.array(X)
    X_mod = np.insert(X_mod, 0, 1, axis=1)
    w_mod = np.insert(w, 0, b, axis=1)  # w[0] is b vector 
    """
    
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################
            """
            z = np.dot(w_mod, X_mod[n].reshape(
                1, D+1).transpose())  # C x 1 Matrix
            prob_classes = softmax1(z)
            prob_classes[y[n]] = prob_classes[y[n]] - \
                1  # for case when y = y{n}
            w_mod = w_mod - step_size * \
                np.dot(prob_classes, X_mod[n].reshape(1, D+1))
            """  
            
            yn = np.identity(C)[y[n]]
            z = np.dot(w, X[n].reshape(
                1, D).transpose()).reshape(C,) + b  # C x 1 Matrix
            prob_classes = softmax1(z)
            error = prob_classes - yn  # (C,)
            w_del = np.dot(error.reshape(C, 1),
                           X[n].reshape(1, D))  # (C,).(D,)
            b_del = error
            w = w - step_size * w_del
            b = b - step_size * b_del

        # return w and b
        #w, b = w_mod[:, 1:], w_mod[:, 0]
        
        

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        
        y = np.identity(C)[y]
        """
        for it in range(max_iterations):
            z = np.dot(X_mod, w_mod.transpose())  # (N, D+1) x (D+1, C) = (N,C)
            preds = softmax(z)
            error = preds - y
            w_mod = w_mod - (step_size / N) * \
                np.dot(error.transpose(), X_mod)  # (C, D+1) = (C, N) * (N, D+1)
        
        # return w and b
        w, b = w_mod[:, 1:], w_mod[:, 0]
        """
        
        for _ in range(0, max_iterations):
            z = (np.dot(w, X.T)).T + b  # (N,C)
            preds = softmax(z)
            error = preds - y
            w_del = np.dot(error.T, X) / N
            b_del = np.mean(error, axis=0)
            w = w - step_size * w_del
            b = b - step_size * b_del
        

    else:
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    def softmax(x):
        x = np.exp(x - np.amax(x))
        denom = np.sum(x, axis=1)
        return (x.T / denom).T

    z = softmax((w.dot(X.T)).T + b)
    #z = np.dot(X, w.transpose()) + b # both are same as exponential is also an increasing function
    preds = np.argmax(z, axis=1)

    
    assert preds.shape == (N,)
    return preds




        