# kaggle-titanic
Predictive analysis of the [titanic dataset](https://www.kaggle.com/c/titanic) on Kaggle.

Features were engineered using the approach in the tutorials. Missing values for the 'Embarked' column were imputed by looking at the most popular embarked location from the passenger's Pclass (there were only two missing values, both females from Pclass 1). The first model tested was a random forest. Note that I chose not to use the model "myfirstforest.py", instead analyzing validation accuracy based on modyfing parameters. Beginning with the number of trees using the entropy function:

![Figure1](https://github.com/faisale/kaggle-titanic/blob/master/images/tree_accuracy_entropy.png?raw=true)

We can see that the accuracy saturates extremely fast, somewhere around 90 trees, although the noise decreases afterwards. Next up was testing the accruacy using the gini function:

![Figure2](https://github.com/faisale/kaggle-titanic/blob/master/images/tree_accuracy_gini.png?raw=true)

The difference between the two functions was insignificant, so I went with the entropy function as it has the highest accuracy score at 185 trees. Afterwards, I tested the number of features to include in each decision tree (the x-axis should say features, not trees!):

![Figure3](https://github.com/faisale/kaggle-titanic/blob/master/images/number_of_features.png?raw=true)

Best performance was tied with five, six and eight features. I went ahead with five. Finally, testing the minimum number of leafs per tree brought interesting results (again, the x-axis should say features, not trees!):

![Figure4](https://github.com/faisale/kaggle-titanic/blob/master/images/min_leaf_sample.png?raw=true)

It seems that the accuracy sharply declines as the minium number of leaf increases. This could be due to the small amount of features and the simplicity of the feature relationships.

Using a random forest with 185 trees, 5 features and a minimum leaf sample of 2 brought an accuracy of 0.76555. We'll come back to this later, but for now, let's analyze the gender survival count:

![Figure5](https://github.com/faisale/kaggle-titanic/blob/master/images/gender_survival.png?raw=true)

The (gender, surivived) format is used where males are 0 and females are 1. Surived is encoded as 1, while death is 0. We can use a simple model to predict the survival of each passenger by predicting that females survive, and males don't. This yields an accuracy of 0.76555 on the test data, the exact same accuracy as the random forest model above. This may tell us that the random forest model places heavy emphasis on gender to predict survival.

Lastly, a logistic regression function was used. With L1 regularization, once again an accuracy of 0.76555 was achieved. However, using L2 regularization brought an improved score of 0.77512!
