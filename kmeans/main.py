import kmeans as km
import numpy as np


def main():
    data = np.loadtxt("cluster_dataset.txt")
    model = km.KMeans(data)

    print(f"\n\n\n{data=}")



if __name__ == "__main__":
    main()


