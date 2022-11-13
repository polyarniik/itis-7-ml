import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=903
        )

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, k):
        predictions = []
        for x in self.X_test:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_nearest_labels = [self.y_train[i] for i in np.argsort(distances)[:k]]
            most_common = Counter(k_nearest_labels).most_common()
            predictions.append(most_common[0][0])
        return predictions

    def get_accuracy(self, predictions):
        return np.sum(predictions == self.y_test) / len(self.y_test)


if __name__ == '__main__':
    knn = KNN()

    while True:
        try:
            k = int(input('Enter k: '))
        except ValueError:
            print(f'Enter a number! {k} is not a number.')
            continue

        if k < 0:
            print('Bye!')
            exit()

        predictions = knn.predict(k)
        print(knn.get_accuracy(predictions))
