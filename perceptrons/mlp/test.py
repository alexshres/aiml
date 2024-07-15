import numpy as np

def main():

    X = np.random.uniform(0, 1, (3, 4))

    first_row = X[:, 0]

    print(f"Dimensions of X is {X.shape} and dimension of first row is {first_row.shape}")


main()
