import numpy as np
import struct

def parse_images(filename):
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    sx,sy = struct.unpack('>ii', f.read(8))
    X = []
    for i in range(size):
        im =  struct.unpack('B'*(sx*sy), f.read(sx*sy))
        X.append([float(x)/255.0 for x in im]);
    return np.array(X);

def parse_labels(filename):
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], 
                                    dtype=np.float64)
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    return one_hot(np.array(struct.unpack('B'*size, f.read(size))), 10)

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]

# helper functions for loss
# this returns a tuple (softmax loss, softmax loss gradient)
softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y), 
                              np.exp(yp)/np.sum(np.exp(yp)) - y)

def get_errors(yp, y, ypt, yt):
    """
    Helper function to compute errors and losses
    
    Arguments:
        yp: m x 10 numpy array of outputs from the classifer
        y: m x 10 numpy array of the true outputs
        ypt: m x 10 numpy array of outputs from the classifer for test data
        yt: m x 10 numpy array of the true outputs for test data
        
    Return:
        (train_err, train_loss, test_err, test_loss)
    """
    train_err = error(yp, y)
    test_err = error(ypt, yt)
    train_loss = sum(softmax_loss(yp[i],y[i])[0] 
                     for i in range(y.shape[0])) / y.shape[0]
    test_loss = sum(softmax_loss(ypt[i],yt[i])[0] 
                    for i in range(yt.shape[0])) / yt.shape[0]
    return (train_err, train_loss, test_err, test_loss)

# function calls to load data
#X_train = parse_images("train-images.idx3-ubyte")
#y_train = parse_labels("train-labels.idx1-ubyte")
#X_test = parse_images("t10k-images.idx3-ubyte")
#y_test = parse_labels("t10k-labels.idx1-ubyte")

##### Implement the functions below this point ######

def grad(Theta, x, y):
    """ 
    Compute the gradient given input x and output y, and current parameters Theta
    Note that this assumes the constant 1 has already been appended to the input
    
    Arguments:
        Theta: 10 x 785 numpy array of current parameters
        x: 785 sized 1D numpy array of input
        y: 10 sized 1D numpy array of output
        
    Return:
        A 10 x 785 numpy array gradient
    """
    grad_array = np.zeros((10, 785));
    
    Xt = np.reshape(x, (785, 1))
    
    L, grad = softmax_loss(Theta.dot(x), y)
    
    
    grad_array = grad*Xt
    
    
    return grad_array.T


def softmax_sgd(X,y, Xt, yt, epochs=10, alpha = 0.01):
    """ 
    Run stochastic gradient descent to solve linear softmax regression.
    
    Arguments:
        X: numpy array where each row is a training example input of length 784
        y: numpy array where each row is a training output of length 10
        Xt: numpy array of testing inputs
        yt: numpy array of testing outputs
        epochs: number of passes T to make over the whole training set
        alpha: step size
        
    Return:
        A list of tuples (Train Err, Train Loss, Test Error, Test Loss) for each epoch
        These should be computed at the end of each epoch
    """
    o1 = np.ones((len(X),1))
    X = np.concatenate((X, o1), axis=1)
    
    o2 = np.ones((len(Xt),1))
    Xt = np.concatenate((Xt, o2), axis=1)
    
    Theta = np.zeros((10, 785));
    
    output = []
    
    for t in range(epochs):
        for i in range(len(X)):
            
            Theta = Theta - alpha * grad(Theta, X[i], y[i])
        
        output.append(get_errors(X.dot(Theta.T), y, Xt.dot(Theta.T), yt))
    
    return output
