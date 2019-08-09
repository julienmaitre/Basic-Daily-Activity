import numpy as np
from sklearn.decomposition import PCA

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

pca = PCA(n_components=50)

pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)

datasetFileName = 'dataset_V1.pickle'
print(datasetFileName)

with open(datasetFileName, 'wb') as f:
    pickle.dump([X_train, y_train, X_test, y_test, IDPeopleForTraining, IDPeopleForTesting], f)

    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))