import kmeans as km
import numpy as np


def main():
    data = np.loadtxt("cluster_dataset.txt")



    for c in range(2, 10):
        model = km.KMeans(data, centers=c)
        print(f"{c=}")
        model.train()



if __name__ == "__main__":
    main()


