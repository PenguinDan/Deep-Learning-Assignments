import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0;
    dW = np.zeros_like(W);

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #Total loss added for each class classification
    total_loss = 0;
    #Get scores for each class where each value in the row is equal to
    #W(j)*X(i), now we can do e^(W(j)*X(i)) for each value in the row
    for i in range(batch_size):
        curr_x = X[i];
        #Does a 1xD * DxC to output a vector 1xC with scores for each class
        ind_class_scores = np.matmul(curr_x, W);
        #Get the max value from the current row normalize the vector
        row_max_value = np.max(curr_x);
        #Subtract the max value from the current row so that the highest
        #Value found in the current row is 0 and everything else is negative
        curr_x = curr_x - row_max_value;
        #Begin the softmax formula of
        #e^(individual class score) / Summation(e^(Each class score in the row))
        #e_ind_class_scores = e^(individual class scores)
        e_ind_class_scores = np.exp(ind_class_scores);
        #softmax_output will now carry a percentage score for each value in the
        #row in respect to their classes
        #softmax_output = e^(individual class scores) / Summation(e^(Each class score in the row))
        softmax_output = e_ind_class_scores / np.sum(e_ind_class_scores);
        #According to the Stanford Lectures, the Loss function is just the
        #Natural Log of the softmax_output
        #-ln(e^(individual class scores) / (Sum of class scores)) = Loss Function
        #In this case, the "Loss" is in respect to its correct class classification
        #So we modify the formula to be -ln(e^(score of class Y) / Summation(e^(Each class score in the row)))
        label = y[i];
        log_class_score = -np.log(softmax_output[label]);
        #Add the loss value received from classifying class Y incorrectly
        total_loss += log_class_score;
        #According to the website https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        #The gradient of the softmax_outputs work in the following manner
        #Gradient of Cross Entropy-Softmax = (Sj - DELTA)*Xi
        #Where DELTA = 1 if j == Class Label, Sj = the value from the softmax vector with index j,
        #Xi = the current row from the X matrix
        #In the inner loop, iterate through the total amount of classifications/softmax vector values
        for j in range(W.shape[1]):
            #Get the current softmax value from the softmax vector
            softmax_val = softmax_output[j];
            #0 if j != Correct Class Label, 1 if j == Correct Class Label
            delta = 0;
            if j == label:
                delta = 1;
            #Update the current gradient with (Sj - DELTA) * X[i]
            dW[:, j] += (softmax_val - delta)*X[i];
    #Divide total_loss variable by batch_size to get the average loss
    avg_loss = total_loss/batch_size;
    #Add the regularization value to the average loss to get the total loss
    loss = avg_loss + reg*np.sum(np.dot(W.T, W));
    #Divide total gradient by batch_size to get the average gradient
    dW = (dW/batch_size);
    #Add the regularization value which is the Gradient in respect to W of
    #lambda*W.T*W which is 2*lambda*W
    dw = dw + 2*reg*W;

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    #Return average loss and average gradient
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #
    #Get scores for each class where each value in the row is equal to
    #W(j)*X(i), now we can do e^(W(j)*X(i)) for each value in the row
    #Where X = N x D and W = D x C where K is the number of classes
    class_scores = np.matmul(X, W);
    #First get the maximum value of each row in a N x 1 vector
    row_max_values = np.amax(class_scores, axis=1);
    #First reshape the row maxd values to be able to subtract the matrix
    batch_size = X.shape[0];
    row_max_values = row_max_values.reshape(batch_size, 1);
    class_scores = class_scores - row_max_values;
    e_class_scores = np.exp(class_scores);
    #Get the total added value from each row of the eClassScores variable
    #axis=1 forces the sum to be carried accross each row
    row_sum_values = np.sum(e_class_scores, axis=1, keepdims=True);
    #Now get the probability values for each class score so that a whole row = 1
    softmax_outputs = e_class_scores / row_sum_values;
    #Now get the -log value of each row according to the proper class y, where y
    #Acts as the index value to choose the proper column in each row
    row_loss = -np.log(softmax_outputs[range(batch_size), y]);
    #Add up all of the class loss values together, and take the average
    total_loss = np.sum(row_loss);
    avg_loss = total_loss / batch_size;
    #Get the regularization value in the proper form, reg*(W^2)
    reg_value = reg*np.sum(np.dot(W.T,W));
    #Final loss is equal to the averages total loss throughout each class + regularization
    loss = avg_loss + reg_value;
    #Now calculate the gradient
    #According to the formula of (SoftmaxFunctionValue - p)x where p = 1 if j == label
    #and p = 0 if j != label, update the softmax_ouputs matrix
    #The following chooses each row and the corresponding y[k] index to update with -1
    softmax_outputs[range(batch_size) , y] -= 1;
    #X.T = DxN and softmax_ouputs = NxC, Weights are DxC, so same format
    dW = np.matmul(X.T, softmax_outputs);
    #Get the average over every weight value
    dW = dW/batch_size;
    #Add the regularization value to every weight value
    dW = dW + 2*reg*W;
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
