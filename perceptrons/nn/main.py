import two_layer_MLP as tm
import mnist_data as md
import numpy as np
import nn 



def main():
    """
    # data paths
    train_images_path = './data/mnist_gz/train-images-idx3-ubyte.gz'
    train_labels_path = './data/mnist_gz/train-labels-idx1-ubyte.gz'

    train_data = md.load_images(train_images_path)
    train_labels = md.load_labels(train_labels_path)

    test_images_path = './data/mnist_gz/t10k-images-idx3-ubyte.gz'
    test_labels_path = './data/mnist_gz/t10k-labels-idx1-ubyte.gz'

    test_data = md.load_images(test_images_path)
    test_labels = md.load_labels(test_labels_path)
    """

    # XOR example
    X = np.array([[0, 0], [0, 1], [1, 0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nodes = [2, 2, 2]


    model = nn.NeuralNetwork(X, y, nodes=nodes)

    model.train()
    
    return


if __name__ == "__main__":
    main()
   
