import mlp
import numpy as np


def main():
    # XOR example
    X = np.array([[0, 0], [0, 1], [1, 0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nodes = [2, 2, 2]


    print(f"Dataset is {X}")
    model = mlp.MLP(X, y, nodes=nodes)

    model.train()


    return

if __name__ == "__main__":
    main()
   
