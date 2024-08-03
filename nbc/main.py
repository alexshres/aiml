import pandas as pd
import spam_data as sd
import nbc

from sklearn.model_selection import train_test_split


def main():
    # Naive Bayes classifier will take in a pandas dataframe object
    X_train, X_test, y_train, y_test = train_test_split(
                                            sd.features, 
                                            sd.target, 
                                            test_size=0.5, 
                                            stratify=sd.target      # Ensures 60/40 split
                                        )

    print(sd.features.columns)


if __name__ == "__main__":
    main()
