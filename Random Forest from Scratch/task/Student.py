import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from matplotlib import pyplot as plt

np.random.seed(52)


def create_bootstrap(x, y, size):
    inds = np.random.choice(y.shape[0], size)
    return x[inds], y[inds]


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error

        self.forest = [DecisionTreeClassifier(max_depth=max_depth, max_features='sqrt', min_impurity_decrease=min_error)
                       for _ in range(n_trees)]
        self.is_fit = False

    def fit(self, x_train, y_train):
        # Your code for Stage 3 here
        for tree in self.forest:
            xb, yb = create_bootstrap(x_train, y_train, y_train.shape[0])
            tree.fit(xb, yb)
        self.is_fit = True

    def predict(self, x_test):
        if not self.is_fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')
        # Your code for Stage 4 here
        values = np.zeros(len(x_test))
        for tree in self.forest:
            values += tree.predict(x_test)
        predictions = [1 if val > self.n_trees / 2 else 0 for val in values]
        return predictions


def stage1(x: np.array, y: np.array, x_test: np.array, y_test: np.array) -> None:
    dtc = DecisionTreeClassifier()
    dtc.fit(x, y)
    print(f'{dtc.score(x_test, y_test):.3f}')


def stage2(x: np.array, y: np.array) -> None:
    xb, yb = create_bootstrap(x, y, 52)
    print(list(yb[:10]))


def stage3(x: np.array, y: np.array, x_test: np.array, y_test: np.array) -> None:
    rfc = RandomForestClassifier()
    rfc.fit(x, y)
    print(f'{rfc.forest[0].score(x_test, y_test):.3f}')


def stage4(x: np.array, y: np.array, x_test: np.array, y_test: np.array) -> None:
    rfc = RandomForestClassifier()
    rfc.fit(x, y)
    print(rfc.predict(x_test)[:10])


def stage5(x: np.array, y: np.array, x_test: np.array, y_test: np.array) -> None:
    rfc = RandomForestClassifier()
    rfc.fit(x, y)
    print(round(accuracy_score(y_test, rfc.predict(x_test)), 3))

def stage6(x: np.array, y: np.array, x_test: np.array, y_test: np.array) -> None:

    scores = []
    max_trees = 21
    for n_trees in tqdm(range(1,max_trees)):
        rfc = RandomForestClassifier(n_trees=n_trees)
        rfc.fit(x, y)
        scores.append(round(accuracy_score(y_test, rfc.predict(x_test)), 3))
    print(scores[:20])
    #plt.scatter(range(1,max_trees), scores)
    #plt.show()


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = \
        train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    # Make your code here...
    # stage1(X_train, y_train,X_val,y_val)
    # stage2(X_train, y_train)
    # stage3(X_train, y_train, X_val, y_val)
    # stage4(X_train, y_train, X_val, y_val)
    stage6(X_train, y_train, X_val, y_val)
