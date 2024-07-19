import two_layer_MLP as tm
import mnist_data as md
import numpy as np
import mlp 

# data paths
train_images_path = './data/mnist_gz/train-images-idx3-ubyte.gz'
train_labels_path = './data/mnist_gz/train-labels-idx1-ubyte.gz'

train_data = md.load_images(train_images_path)
train_labels = md.load_labels(train_labels_path)

test_images_path = './data/mnist_gz/t10k-images-idx3-ubyte.gz'
test_labels_path = './data/mnist_gz/t10k-labels-idx1-ubyte.gz'

test_data = md.load_images(test_images_path)
test_labels = md.load_labels(test_labels_path)

def experiment_1():
    for n in [20, 50, 100]:
        model = tm.TwoMLP(train_data, train_labels, hidden_nodes=n,
                          outputs=10, epochs=50, 
                          learning_rate=0.1, momentum=0.9,
                          type="hidden_nodes", type_value=n)

        model.train(test_data, test_labels)
        model.fit(test_data, test_labels)

def experiment_2():
    for m in [0, 0.25, 0.5]:
        model = tm.TwoMLP(train_data, train_labels, hidden_nodes=100,
                          outputs=10, epochs=50, 
                          learning_rate=0.1, momentum=m,
                          type="momentum", type_value=m)

        model.train(test_data, test_labels)
        model.fit(test_data, test_labels)


def experiment_3():
    for s in [0.25, 0.5]:
        size = int(s * train_data.shape[0])

        p = np.random.permutation(len(train_data))
        new_train_data = train_data[p]
        new_train_labels = train_labels[p]

        new_train_data = new_train_data[:size, :] 
        new_train_labels = new_train_labels[:size]

            
        model = tm.TwoMLP(train_data, train_labels, hidden_nodes=100,
                          outputs=10, epochs=50, 
                          learning_rate=0.1, momentum=0.9,
                          type="training_size", type_value=size)

        model.train(test_data, test_labels)
        model.fit(test_data, test_labels)



def main():
    """
    experiment_1()
    experiment_2()
    experiment_3()

    """

    # XOR example
    X = np.array([[0, 0], [0, 1], [1, 0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nodes = [2, 2, 2]


    model = mlp.MLP(X, y, nodes=nodes)

    model.train()
    
    return


if __name__ == "__main__":
    main()
   
