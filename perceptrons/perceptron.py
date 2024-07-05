import numpy as np


class Perceptron:
    def __init__(self, X: np.ndarray, y: np.array):
        self.X = X
        self.y = y

        # Adding bias column to each training data
        ones_column = np.ones((X.shape[0], 1))

        self.X = np.hstack((ones_column, self.X))

        self.learning_rate = 0.1

        # Initializing our weights to be all zeroes
        # Size of weights is the number of rows in our input array
        self.weights = np.zeros((self.X.shape[1], 1))

        self.iters = 100


    def fit(self):
        predicted = self.X @ self.weights

        update_weights = np.zeros((self.X.shape[1], 1))
        
        for i in range(self.iters):
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]

            # For each set of data (for each row)
            for i in range(self.X.shape[0]):
                # Checking if signs match
                predict_sign = np.sign(predicted[i])
                actual_sign = np.sign(self.y[i])

                # If our prediction is incorrect
                if predict_sign != actual_sign:
                    # For each weight coefficient we calculate update value
                    # using stochastic gradient descent
                    for j in range(update_weights.shape[0]): 
                        update_weights[j] = self.X[i, j] * (np.dot(self.X[i], self.weights) - self.y[i])
               # If our prediction is correct, just set update_weights to all zeroes
                else:
                    update_weights = 0 * update_weights

                # Updating weights with whatever the learning_rate is
                self.weights = self.weights - (self.learning_rate * update_weights)

        return self.weights




def ols(A: np.ndarray, b: np.array) -> np.ndarray:  # returns best fit line
    x = np.zeros(A.shape[0])
    A_t = np.linalg.matrix_transpose(A)
    inv_sym = np.linalg.inv(A_t @ A)         # matmul
    half_proj = inv_sym @ A_t
    x = half_proj @ b

    return x




def main():

    A = np.array([[2, 3], [1, 1],
                  [2, 1], [1, 2]])
    b = np.array([1, -1, 1, -1])

    perceptron = Perceptron(A, b)
    weights = perceptron.fit()

    print(weights)

    return


if __name__ == "__main__":
    main()
