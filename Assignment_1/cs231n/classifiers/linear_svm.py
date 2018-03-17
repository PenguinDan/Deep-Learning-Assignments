import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    #Contains the current shape D x C
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #num_classes = C
    num_classes = W.shape[1]
    #num_train = N
    num_train = X.shape[0]
    #Initialize the current value for loss to 0
    loss = 0.0
    #Initalize delta for the SVM loss function
    delta = 1

    #Begin calculations
    for i in xrange(num_train):
        #Retrieve the current example
        curr_example = X[i]
        #Calculate the scores for each class for the current example,
        #scores contains the confidence value for each class with the shape 1 x C
        scores = np.matmul(curr_example, W)
        #Retrieve the correct label
        label = y[i]
        #Retrieve the class score of the correct label
        correct_class_score = scores[label]
        #Count the number of times that the loss becomes greater than delta to
        #update correct class gradient weights
        incorrect_classification = 0
        #Calculate the SVM loss value excluding the correct class label
        for j in xrange(num_classes):
            #Continue if j == our label, we don't care about it for SVM
            if j == label:
                continue
            #SVM formula, Max(0, Sj - Correct_class_score + Delta) where
            #Delta = 1
            loss_val = scores[j] - correct_class_score + delta
            #Loss value is greater than 0, also calculate the gradient
            #inside below statement, the gradient can be seen from the study notes
            #total_derivative = sum(I(something - w_y*x[i] > 0) * (-x[i]))
            #Where the first part is 0 if j != label and 1 if j == label, however
            #The above is the gradient, negate it to get the gradient descent formula
            if loss_val > 0:
                #Update the incorrect_classification value to update correct class
                #gradient once we break out of the loop
                incorrect_classification += 1
                loss += loss_val
                #Grab the column representing the weights for the class
                dW[:, j] += curr_example
        #Now that we are outside of the loop, update the gradient in respect of the
        #correct label weight
        dW[: , label] += -incorrect_classification*curr_example

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss = loss / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    #Get the average gradient loss
    dW = (dW / num_train) + 2*reg*W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #Initialize the current value for loss to 0
    loss = 0.0
    #Contains the current shape D x C
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #The number of examples in X
    number_of_examples = X.shape[0]
    #The number of classes in W
    number_of_classes = W.shape[1]
    #Initalize delta for the SVM loss function
    delta = 1

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    #We want an N x C matrix in the end where each row represents the scores for
    #each class relative to the row example in X and the column values in W
    #score_matrix = (N x D)(D x C) = (N x C)
    score_matrix = np.matmul(X, W)

    #Use numpy's choose method to pluck out specific values that correspond
    #to each element in y vector from each row in the score_matrix
    #We must transpose score_matrix to C x N because np.choose works column wise
    correct_class_score_matrix = np.choose(y, score_matrix.T)
    #Subtract 1 from the correct_class_score_matrix so that we dont have to add
    #1 later to the larger matrix
    correct_class_score_matrix -= 1

    #Create a matrix where each element of correct_class_score_matrix repeats
    #itself over an entire row for an N x C matrix

    #First reshape the correct_class_score_matrix from a 1 x N matrix to an N x 1
    #matrix
    correct_class_score_matrix = correct_class_score_matrix.reshape(number_of_examples, 1)
    #Extend the N x 1 matrix to an N x C using broadcasting using a zero matrix
    correct_class_score_matrix = np.zeros((number_of_examples, number_of_classes)) + correct_class_score_matrix
    #Now correct_class_score_matrix is an N x C matrices

    #Turn the values of each row element corresponding with the label value associated
    #with the example into 0
    for i in range(number_of_examples):
        label = y[i]
        correct_class_score_matrix[i][label] = 0
        score_matrix[i][label] = 0

    loss = score_matrix - correct_class_score_matrix
    loss = np.sum(loss[np.where(loss > 0)])
    #Get the average loss
    loss = loss/number_of_examples
    #Add regularizer
    loss = loss + reg*np.sum(W*W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    #Retrieve the a mask of the locations where the loss value is greater than 0
    #and apply it to the X matrix
    #Following technique learned from
    #https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
    mask = ((score_matrix-correct_class_score_matrix) > 0).astype(int)
    #Mask now holds a matrix of values of 1 and 0, multiple the two matrices
    #(D x N) (N x C) = D x C
    dW += np.matmul(X.T, mask)

    #Get the sum of the mask row wise to add up all of the times that a value
    #greater than 0 was found per example
    summed_mask = np.sum(mask, axis= 1)
    #Transpose the summed mask, however do it throug the reshape so it doesnt
    #turn into a list
    summed_mask = summed_mask.reshape(number_of_examples, 1)
    #Broadcast the summed_mask values accross the X array so that every row example
    #is multiplied by how many times that row received a SVM loss of greater than 0
    masked_X = X*summed_mask

    #Can't think of a way to do it with just a matrix
    for j in range(len(y)):
        label = y[j]
        dW[:, label] -= masked_X[j]

    dW = dW / number_of_examples
    dW = dW + 2*reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
