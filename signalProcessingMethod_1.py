import os
import time

import pandas
import numpy as np

from sklearn.model_selection import train_test_split

import pickle

import featureExtraction

########################################################################################################################
#                                                    User Parameters                                                   #
########################################################################################################################

# Path name of the signal files directory (Provided by the user)
pathName = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\Signaux Total"

# List of objects held by the hand and gestures of the hand (Provided by the user)
objectList = ['Fourchette', 'Cuillere', 'Couteau', 'Assiette', 'Verre', 'Tasse', 'Casserole', 'Poele', 'Fouet',
              'Louche', 'PressoirePatates', 'Cafetiere', 'Rouleau', 'MainAVide']

# Parameters of the sliding windows to extract features (signal processing)
timeWindowsLength = 20
overlappingLength = 3

# Ratio of the testing dataset
ratioTestingDataset = 0.3

########################################################################################################################
#                                                    Initialization                                                    #
########################################################################################################################

# List the number of folders in the pathname provided by the user
folderList = os.listdir(pathName)  # the folders are those with the person id or name

# Compute the number of folders
folderNumber = len(folderList)

########################################################################################################################
#                                                   Create Datasets                                                    #
########################################################################################################################

# Initialize the label variable for the dataset
yData = np.array([])
IDPeopleForTraining = []
IDPeopleForTesting = []

total_time = 0

# For each first n people in the permutation
for folderName in folderList:

    tmpPathName = os.path.join(pathName, folderName)
    fileList = os.listdir(tmpPathName)
    IDPeopleForTraining.append(folderName)
    IDPeopleForTesting.append(folderName)

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

                    if 'xData' in locals():

                        xData = np.concatenate((xData, np.array([tmp])), axis=0)

                    else:

                        xData = np.array([tmp])

                    yData = np.append(yData, object)

X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=ratioTestingDataset, random_state=42)

datasetFileName = 'dataset.pickle'
print(datasetFileName)

with open(datasetFileName, 'wb') as f:
    pickle.dump([X_train, y_train, X_test, y_test, IDPeopleForTraining, IDPeopleForTesting], f)

    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))

print("\n\nThe total time consuming to extract the features are : %.8f second", total_time)

