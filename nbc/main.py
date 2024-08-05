import pandas as pd
import nbc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from ucimlrepo import fetch_ucirepo

def main():
    spambase = fetch_ucirepo(id=94)
    features = spambase.data.features
    target = spambase.data.targets

    # Naive Bayes classifier will take in a pandas dataframe object
    X_train, X_test, y_train, y_test = train_test_split(
                                            features, 
                                            target, 
                                            test_size=0.5, 
                                            stratify=target      # Ensures 60/40 split
                                        )

    classifier = nbc.NaiveBayesBinary(training_data=X_train, 
                                      training_labels=y_train, 
                                      num_classes=2)

    
    y_pred = classifier.predict(X_test)
    targets = y_test['Class'].tolist()


    accuracy = accuracy_score(targets, y_pred)
    precision = precision_score(targets, y_pred)
    recall = recall_score(targets, y_pred)
    conf_matrix = confusion_matrix(targets, y_pred)


    print(f"{accuracy=}\n{precision=}\n{recall=},\n{conf_matrix=}")



if __name__ == "__main__":
    main()
