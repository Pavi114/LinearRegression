import numpy as np

#########################
# Class to fit a data-set into a straight line
# And to predict output for new values
##########################


class LinearRegression:

    # constructor
    def __init__(self, X, Y, iterations=1000, learn_rate=0.005):
        self.num_samples = len(Y)
        self.num_features = X.shape[1]
        self.X = X
        self.Y = Y
        self.parameters = np.zeros([self.num_features + 1, 1])
        self.num_iterations = iterations
        self.cost_history = np.zeros(self.num_iterations)
        self.learn_rate = learn_rate

    ##############################
    # To normalize data (x - mu) / sd
    # Input: Data to be normalised, Number of samples
    # Output: Normalized Data along with one extra column of ones for bias
    #############################

    def normalize(self, x, n_samples):
        data = (x - np.mean(x, 0)) / np.std(x,0)
        return np.hstack((np.ones([n_samples, 1]), data))

    #############################
    # To Calculate the predicted output
    # Output: X * parameters
    #############################

    def hypothesis(self):
        return np.dot(self.X, self.parameters)

    #############################
    # To calculate the cost using the formula
    # cost = 1/2 * mean squared error
    # Output: cost
    #############################

    def compute_cost(self):
        num_samples = len(self.Y)
        predicted_value = self.hypothesis()
        return 1/(2 * num_samples) * np.sum(np.square(predicted_value - self.Y))

    #####################################
    # Training Data using Gradient Descent
    # Computing parameters to fit the given the data set into a straight line
    # Output: self object
    ######################################

    def fit(self):
        self.X = self.normalize(self.X, self.num_samples)
        for iteration in range(self.num_iterations):
            self.parameters -= self.learn_rate * (self.X.T.dot(self.hypothesis() - self.Y) / self.num_samples)
            self.cost_history[iteration] = self.compute_cost()
        return self

    ######################################
    # Predict value of Y using test data
    # Input: Test Data-set
    # Output: Predicted Y for the new data-set
    #####################################

    def predict(self, x_test):
        x_normalize_x = self.normalize(x_test, np.shape(x_test)[0])
        y_predict = np.dot(x_normalize_x, self.parameters)
        return y_predict
