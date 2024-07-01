import numpy as np


def ols(A: np.ndarray, b: np.array) -> np.ndarray:  # returns best fit line
    x = np.zeros(A.shape[0])
    A_t = np.linalg.matrix_transpose(A)
    inv_sym = np.linalg.inv(A_t @ A)         # matmul
    half_proj = inv_sym @ A_t
    x = half_proj @ b

    return x




def main():

    A = np.array([[3, 2], [1, 0]], np.int32)
    b = np.ones(2)
    
    y = ols(A, b)

    print(y)

    return




if __name__ == "__main__":
    main()

    



