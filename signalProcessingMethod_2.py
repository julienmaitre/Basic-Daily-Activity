import os
import time

import pandas
import numpy as np
from itertools import combinations

import pickle

import featureExtraction


def missingValues(a,b):

    values = []

    for i in a:

        if i not in b:

            values.append(i)

    return values

########################################################################################################################
#                                                    User Parameters                                                   #
########################################################################################################################

# Path name of the signal files directory (Provided by the user)
pathName = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\Signaux Total"

# List of objects held by the hand and gestures of the hand (Provided by the user)
objectList = ['Fourchette', 'Cuillere', 'Couteau', 'Assiette', 'Verre', 'Tasse', 'Casserole', 'Poele', 'Fouet',
              'Louche', 'PressoirePatates', 'Cafetiere', 'Rouleau', 'MainAVide']

# Number of dataset corresponding to people to leave for the testing process
peopleNumberLeave = 1

# Length of the time window in samples
timeWindowsLength = 20

# Size of the overlapping in samples
overlappingLength = 3

########################################################################################################################
#                                                    Initialization                                                    #
########################################################################################################################

# List the number of folders in the pathname provided by the user
folderList = os.listdir(pathName)  # the folders are those with the person id or name

# Compute the number of folders
folderNumber = len(folderList)

# Permutation to create xData and yData
combinationResults = combinations(range(folderNumber),folderNumber-peopleNumberLeave)

########################################################################################################################
#                                                   Create Datasets                                                    #
########################################################################################################################

count = 1
total_time = 0

# For each permutation of indices corresponding to the person id
for i in list(combinationResults):

    # Initialize the label variable for the dataset
    y_train = np.array([])
    IDPeopleForTraining = []

    # For each first n people in the permutation
    for j in range(0, len(i)):

        index = i[j]
        tmpPathName = os.path.join(pathName, folderList[index])
        fileList = os.listdir(tmpPathName)
        IDPeopleForTraining.append(folderList[index])

        for fileName in fileList:

            for object in objectList:

                if object in fileName:

                    tmpFileName = os.path.join(tmpPathName, fileName)
                    data = pandas.read_csv(tmpFileName, sep=',', skiprows=1, engine='python')

                    data = data.values

                    for k in range(0, len(data)-timeWindowsLength, overlappingLength):

                        X = data[k:k+timeWindowsLength]

                        # C'est ici que je dois ajouter les caracteristique a extraire
                        # tmp = np.mean(X, axis=0)
                        # tmp = np.append(tmp, np.std(X, axis=0))

                        # Get the start time of the feature extraction
                        start_time = time.time()

                        tmp = featureExtraction.feature_extraction(X)

                        # Get the end time of the feature extraction
                        end_time = time.time()

                        total_time = total_time + (end_time-start_time)

                        if 'X_train' in locals():

                            X_train = np.concatenate((X_train, np.array([tmp])),axis=0)

                        else:

                            X_train = np.array([tmp])

                        y_train = np.append(y_train, object)

    # Initialize the label variable for the dataset
    y_test = np.array([])
    IDPeopleForTesting = []

    values = missingValues(range(folderNumber),i)

    # For each first n people in the permutation
    for j in values:

        tmpPathName = os.path.join(pathName, folderList[j])
        fileList = os.listdir(tmpPathName)
        IDPeopleForTesting.append(folderList[j])

        for fileName in fileList:

            for object in objectList:

                if object in fileName:

                    tmpFileName = os.path.join(tmpPathName, fileName)
                    data = pandas.read_csv(tmpFileName, sep=',', skiprows=1, engine='python')

                    data = data.values

                    for k in range(0, len(data)-timeWindowsLength, overlappingLength):

                        X = data[k:k+timeWindowsLength]

                        # tmp = np.mean(X, axis=0)
                        # tmp = np.append(tmp, np.std(X, axis=0))

                        # Get the start time of the feature extraction
                        start_time = time.time()

                        tmp = featureExtraction.feature_extraction(X)

                        # Get the end time of the feature extraction
                        end_time = time.time()

                        total_time = total_time + (end_time - start_time)

                        if 'X_test' in locals():

                            X_test = np.concatenate((X_test,np.array([tmp])),axis=0)

                        else:

                            X_test = np.array([tmp])

                        y_test = np.append(y_test, object)

    datasetFileName = 'dataset_' + str(count) + '.pickle'
    count = count + 1
    print(datasetFileName)
    print(IDPeopleForTraining)
    print(IDPeopleForTesting)
    with open(datasetFileName,'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test, IDPeopleForTraining, IDPeopleForTesting], f)

    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))

    del X_train, y_train, X_test, y_test, IDPeopleForTraining, IDPeopleForTesting

print("\n\nThe total time consuming to extract the features are : %.8f second", total_time)


