import numpy as np
from scipy import stats
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
        consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
        y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
        of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
        between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training
        point.
        """
        #The number of test examples in X, N value of the X matrix
        num_test = X.shape[0]
        #The number of training examples, N value of X_train matrix
        num_train = self.X_train.shape[0]
        #Each row is the distance between one test example compared to every training example
        dists = np.zeros((num_test, num_train))
        #Begin calculations
        for i in xrange(num_test):
            #The current testing vector to calculate the difference across every training example
            curr_test_vector = X[i]
            for j in xrange(num_train):
                #The current training vector to be used to get the distance of with the test vector
                curr_train_vector = self.X_train[j]
                #Calculate the L2 Distance between them, we do not use the square root function because
                #It is a monotonic function, meaning it will keep the ordering the same regardless
                distance_vector = np.power(curr_train_vector - curr_test_vector, 2)
                distance = np.sum(distance_vector)
                #Store the calculated distance
                dists[i][j] = distance

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        #The number of test examples, N value of the X matrix
        num_test = X.shape[0]
        #The number of training examples, N value of the X_train matrix
        num_train = self.X_train.shape[0]
        #Each row represents the distance between a single test example compared to
        #every single training example
        dists = np.zeros((num_test, num_train))
        #Begin calculations
        for i in xrange(num_test):
            #The current test vector
            curr_test_vector = X[i]
            #Subtract the test vector from every training row
            distance_matrix = self.X_train - curr_test_vector
            #Calculate the squared power of every value in the distance matrix
            distance_matrix = np.square(distance_matrix)
            #Sum the values of each row together
            distance_vector = np.sum(distance_matrix, axis=1)
            #No need to take the square root because it is a monotonic function
            #Store the distance values in the respective row
            dists[i, :] = distance_vector

        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        #The number of test examples
        num_test = X.shape[0]
        #The number of training examples
        num_train = self.X_train.shape[0]
        #Each row represents the distance between a single test example compared to
        #every single training example
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        #Exand the formula for the L2 Distance formula to the following:
        #Summation of( (X_Train^2) - 2*(X_Train * X_Test) + (X_Test^2) )
        #By following the above formula we get:
        #An N x 1 matrix now
        distance = np.sum(X**2, axis= 1, keepdims= 1)
        #(N x 1) + (1 x Nt), (N x 1) -> (N x Nt), (1 x Nt) -> (N x Nt), end with (N x Nt)
        distance = distance + np.sum(self.X_train**2, axis= 1)
        #In order to get  the right dimensions to return, we formulate it the following way
        #(N x Nt) - 2*[ (N x F)*(F x Nt) ], (N x Nt) - (N x Nt) = (N x Nt)
        distance = distance - 2*np.matmul(X, self.X_train.T)
        #No need to take the square root because it is a monotonic function
        dists = distance
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            ##########################################################################
            #Get the curr distance vector we are working on and store it to make it
            #more readable
            curr_dist_vector = dists[i]
            #Create a sorted array of indices that represent the value of their element
            #from smallest to least
            sorted_indices = np.argsort(curr_dist_vector)
            #Choose the first k values
            smallest_indices_vector = sorted_indices[:k]
            #Retrieve the values from the indices
            closest_y = self.y_train[smallest_indices_vector]
            #closest_y now contains labels that most likely represent the curr train example
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            #Retrieve the label that is the most relevent, if a tie exists, just return
            #the smallest label
            y_pred[i] = stats.mode(closest_y)[0][0]
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred
