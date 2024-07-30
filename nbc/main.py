import pandas as pd
import spam_data as sd
from sklearn.model_selection import train_test_split


def main():
    X_train, X_test, y_train, y_test = train_test_split(
                                            sd.features, 
                                            sd.target, 
                                            test_size=0.5, 
                                            stratify=sd.target      # Ensures 60/40 split
                                        )

    print(X_train.describe())
    print(X_test.describe())
    print(y_train.describe())
    print(y_test.describe())



if __name__ == "__main__":
    main()
