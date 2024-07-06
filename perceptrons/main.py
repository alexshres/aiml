import perceptron as pct
import mnist_data as md

def main():

    """
    A = np.array([[2, 3], [1, 1],
                  [2, 1], [1, 2]])
    b = np.array([1, -1, 1, -1])

    perceptron = Perceptron(A, b)
    weights = perceptron.fit()

    print(weights)
    """


    train_images_path = './mnist_gz/train-images-idx3-ubyte.gz'
    train_labels_path = './mnist_gz/train-labels-idx1-ubyte.gz'

    image_data = md.load_images(train_images_path)
    label_data = md.load_labels(train_labels_path)


    
    print("Printing numpy image data")
    print(image_data)
    print(f"Size of image data is {image_data.shape}")

    print("Printing label data")
    print(label_data)
    print(f"Size of label data is {label_data.shape}")

    perceptron = pct.Perceptron(image_data, label_data)
    weights = perceptron.fit()

    print(weights)

    return


if __name__ == "__main__":
    main()
