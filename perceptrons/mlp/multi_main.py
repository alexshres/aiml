import mlp
import numpy as np


def main():
    X = np.identity(3)
    y = np.ones((3, 1))

    nodes = [785, 20, 10]


    model = mlp.MLP(X, y, nodes=nodes)

    for mat in range(len(model.weights_array)):
        print(model.weights_array[mat].shape)


    return

if __name__ == "__main__":
    main()
   
