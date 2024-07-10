import perceptron as pct
import mnist_data as md
import numpy as np

def main():

    """
    A = np.array([[2, 3], [1, 1],
                  [2, 1], [1, 2]])
    b = np.array([0, 1, 2, 3])

    print(A)
    print(b)

    print(f"Shape of A is {A.shape}")
    print(f"Shape of b is {b.shape}")
    perceptron = pct.Perceptron(A, b, outputs=1)

    weights = perceptron.train()

    print(weights)


    """
    train_images_path = './mnist_gz/train-images-idx3-ubyte.gz'
    train_labels_path = './mnist_gz/train-labels-idx1-ubyte.gz'

    image_data = md.load_images(train_images_path)
    label_data = md.load_labels(train_labels_path)

    test_images_path = './mnist_gz/t10k-images-idx3-ubyte.gz'
    test_labels_path = './mnist_gz/t10k-labels-idx1-ubyte.gz'

    test_data = md.load_images(test_images_path)
    test_labels = md.load_labels(test_labels_path)
    
    print("Printing numpy image data")
    print(image_data)
    print(f"Size of image data is {image_data.shape}")

    print("Printing label data")
    print(label_data)
    print(f"Size of label data is {label_data.shape}")

    inputs = 785
    outputs = 10

    perceptron = pct.Perceptron(inputs=inputs, outputs=outputs, epochs=100)
    weights = perceptron.train(image_data, label_data)

    perceptron.predict(test_data, test_labels)

    return


if __name__ == "__main__":
    main()
