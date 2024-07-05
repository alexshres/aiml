import gzip
import numpy as np
import struct

train_images_path = './mnist_gz/train-images-idx3-ubyte.gz'
train_labels_path = './mnist_gz/train-labels-idx1-ubyte.gz'


def load_images(path: str):
    with gzip.open(path, 'rb') as f:
        _, num, rows, cols = struct .unpack(">IIII", f.read(16))

        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Reshaping data to [num_images, rows*cols)
        images = image_data.reshape(num, rows * cols)

        # Normalizing pixel values to be b/w 0 and 1
        images = images/255.0

        return images


def load_labels(path: str):
    with gzip.open(path, 'rb') as f:

        _, num = struct.unpack(">II", f.read(8))

        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels



