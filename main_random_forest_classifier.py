import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

import random_forest_classification

from sklearn.model_selection import train_test_split

import pickle

with open('dataset_4.pickle', 'rb') as f:
    X_train, y_train,  X_test, y_test, IDPeopleForTraining, IDPeopleForTesting = pickle.load(f)

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

class_names = np.array(['Fourchette', 'Cuillere', 'Couteau', 'Assiette', 'Verre', 'Tasse', 'Casserole', 'Poele', 'Fouet',
              'Louche', 'PressoirePatates', 'Cafetiere', 'Rouleau', 'MainAVide'])

print('People ID for the training dataset :')
print('         %s' % IDPeopleForTraining)
print('People ID for the testing dataset :')
print('         %s' % IDPeopleForTesting)

########################################################################################################################
#                                    User Settings for the Random Forest Classifier                                    #
########################################################################################################################

# Create a class object that define the parameters of the random forest classifier
random_forest_parameters = random_forest_classification.RandomForestParameters()

""" Number of trees in the forest. 
    It should be an integer (and by defaut=10). """
random_forest_parameters.n_estimators = 100

""" Function to measure the quality of a split
    Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. 
    Note: this parameter is tree-specific. """
random_forest_parameters.criterion = 'entropy'

""" The maximum depth of the tree.
    The choices are :
                    If None, then nodes are expanded until all leaves are pure or until all leaves contain less than 
                        min_samples_split samples. """
random_forest_parameters.max_depth = None

""" The minimum number of samples required to split an internal node :
    The choices are :
                      If int, then consider min_samples_leaf as the minimum number.
                      If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are 
                        the minimum number of samples for each node."""
random_forest_parameters.min_samples_split = 2

""" The minimum number of samples required to be at a leaf node.
    The choices are :
                      If int, then consider min_samples_split as the minimum number.
                      If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the 
                        minimum number of samples for each split. """
random_forest_parameters.min_samples_leaf = 1

""" The number of features to consider when looking for the best split
    The choices are :
                      If int, then consider max_features features at each split.
                      If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
                      If “auto”, then max_features=sqrt(n_features).
                      If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
                      If “log2”, then max_features=log2(n_features).
                      If None, then max_features=n_features. 
                      
    Note: the search for a split does not stop until at least one valid partition of the node samples is found, even 
    if it requires to effectively inspect more than max_features features."""
random_forest_parameters.max_feature = 'sqrt'

""" Grow trees with max_leaf_nodes in best-first fashion. 
    If None then unlimited number of leaf nodes. """
random_forest_parameters.max_leaf_nodes = None

""" The number of jobs to run in parallel for both fit and predict. 
    If -1, then the number of jobs is set to the number of CPU cores. 
    The value by default is default=1. """
random_forest_parameters.n_jobs = 1

# Create a class object that define the performances container of the random forest classifier
performances = random_forest_classification.PerformancesRandomForest()

########################################################################################################################
#                                  Execute the Random Forest Classifier on the Dataset                                 #
########################################################################################################################

# Create and train the random forest classifier
random_forest_classifier, training_running_time = \
    random_forest_classification.train_random_forest_classifier(X_train, y_train, random_forest_parameters)

# Print information in the console
print("The training process of random forest classifier took : %.8f second" % training_running_time)

# Test the random forest classifier
y_test_predicted, testing_running_time = \
    random_forest_classification.test_random_forest_classifier(X_test, random_forest_classifier)

# Print information in the console
print("The testing process of random forest classifier took : %.8f second" % testing_running_time)

# Compute the performances of the random forest classifier
cm = random_forest_classification.compute_performances_for_multiclass(y_test, y_test_predicted, class_names, performances)

# Display the results
random_forest_classification.display_confusion_matrix(performances, class_names)
random_forest_classification.display_features_and_classification_for_rf_classifier(X_test, y_test, class_names,
                                                                                   random_forest_classifier,
                                                                                   random_forest_parameters)

plt.show()