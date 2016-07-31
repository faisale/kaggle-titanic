import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn import cross_validation

# Model generation for the titanic data challenge on Kaggle. Multiple models
# were tested and many recieved the same accuracy on the leaderboards.
#
# Author: Faisal El Husseion

# Analysis to fine tune the random forest model.
def RandomForestAnalysis(x_train, x_test, y_train, y_test):
    tree_accuracy = pd.DataFrame(columns=('Trees', 'Accuracy'))

    for i in range(1,51):
        forest = RandomForestClassifier(n_estimators=185,max_features=5,min_samples_leaf=i)
        forest = forest.fit(x_train, y_train)
        accuracy = forest.score(x_test, y_test)

        tree_accuracy.loc[len(tree_accuracy)] = [i, accuracy]
        print(i)

    tree_accuracy.plot(x='Trees', y='Accuracy')
    plt.show()

# Random forest model with tuned parameters.
def RandomForest(x_train, y_train, test_data):
    forest = RandomForestClassifier(n_estimators=185,max_features=5,min_samples_leaf=2)
    forest = forest.fit(x_train, y_train)
    output = forest.predict(test_data)

    GenerateCSV(output, 'random_forest_results.csv')

# Simple logistic regression model with L2 regularization.
def LogisticRegression(x_train, y_train, test_data):
    model = LogisticRegression(penalty='l2', random_state=0)
    model = model.fit(x_train, y_train)
    output = model.predict(test_data)

    GenerateCSV(output, 'logistic_regression_results.csv')

# Gender based model. Females survive, males do not.
def GenderModel(df):
    output = df['Gender']

    GenerateCSV(output, 'gender_model_results.csv')

# Generate the submission file.
def GenerateCSV(output, file_name):
    test = pd.read_csv('csv/test.csv',header=0)

    id = test['PassengerId'].values

    d = {'PassengerId': id, 'Survived': output}

    results = pd.DataFrame(data=d)
    results.to_csv(file_name, index=False)

df = pd.read_csv('train_data.csv', header=0, index_col=0)

train_data = df.drop('Survived', axis=1).values
train_target = df['Survived'].values

x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)

test_df = pd.read_csv('test_data.csv', header=0)
test_data = test_df.values