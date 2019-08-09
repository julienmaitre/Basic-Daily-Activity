import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

import adaboost_classification

import pickle

with open('dataset.pickle', 'rb') as f:
    X_train, y_train, X_test, y_test, IDPeopleForTraining, IDPeopleForTesting = pickle.load(f)

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
#                                      User Settings for the AdaBoost Classifier                                       #
########################################################################################################################

# Create a class object that define the parameters of the random forest classifier
adaboost_parameters = adaboost_classification.AdaBoostParameters()

""" Base estimator from which the boosted ensemble is built. 
    It should be an integer (and by defaut=None - DecisionTreeClassifier(max_depth=1)). """
adaboost_parameters.base_estimator = DecisionTreeClassifier(max_depth=14)

""" Maximum number of estimators at which boosting is terminated.
    In case of perfect fit, the learning procedure is stopped early. 
    It should be an integer (and by default=50) """
adaboost_parameters.n_estimators = 100

""" Learning rate shrinks the contribution of each classifier by learning_rate.
    There is a trade-off between learning_rate and n_estimators.
    It should be a float (and by default=1.) """
adaboost_parameters.learning_rate = 1.

""" Algorithm to boost the algorithm.
    If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class 
    probabilities. If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically 
    converges faster than SAMME, achieving a lower test error with fewer boosting iterations. """
adaboost_parameters.algorithm = "SAMME.R"

""" If int, random_state is the seed used by the random number generator; If RandomState instance, 
    random_state is the random number generator; If None, the random number generator is the 
    RandomState instance used by np.random. """
adaboost_parameters.random_state = None

# Create a class object that define the performances container of the AdaBoost classifier
performances = adaboost_classification.PerformancesAdaBoost()

########################################################################################################################
#                                    Execute the AdaBoost Classifier on the Dataset                                    #
########################################################################################################################

# Create and train the AdaBoost classifier
adaboost_classifier, training_running_time = \
    adaboost_classification.train_adaboost_classifier(X_train, y_train, adaboost_parameters)

# Print information in the console
print("The training process of AdaBoost classifier took : %.8f second" % training_running_time)

# Test the random forest classifier
y_test_predicted, testing_running_time = \
    adaboost_classification.test_adaboost_classifier(X_test, adaboost_classifier)

# Print information in the console
print("The testing process of AdaBoost classifier took : %.8f second" % testing_running_time)

# Compute the performances of the random forest classifier
cm = adaboost_classification.compute_performances_for_multiclass(y_test, y_test_predicted, class_names,
                                                                 performances)

# Display the results
adaboost_classification.display_confusion_matrix(performances, class_names)
adaboost_classification.display_features_and_classification_for_ada_classifier(X_test, y_test, class_names,
                                                                               adaboost_classifier)

plt.show()