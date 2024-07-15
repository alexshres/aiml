import simple_pct as pct
import mnist_data as md
import numpy as np

def simple_pct_main():

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
    
    inputs = 785
    outputs = 10

    for lr in [0.001, 0.01, 0.1]:
        perceptron = pct.Perceptron(inputs=inputs, outputs=outputs, 
                                    learning_rate=lr, epochs=10)
        weights = perceptron.train(image_data, label_data)

        perceptron.predict(test_data, test_labels)

    return

def pytorch_pct_main():
    import pytorch_pct as ptpct

    # Initialize and train the Perceptron
    input_size = 28*28  # 784
    num_classes = 10
    learning_rate = 0.01
    epochs = 10

    model = ptpct.PyTorchPCT(input_size=input_size, num_classes=num_classes, 
                       learning_rate=learning_rate, epochs=epochs)
    model.train_model(ptpct.train_loader)

    # Evaluate the model
    model.evaluate_model(ptpct.test_loader)




if __name__ == "__main__":
    pytorch_pct_main()
