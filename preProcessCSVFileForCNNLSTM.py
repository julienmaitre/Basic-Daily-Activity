""" This script allows generating fake dataset. Each instance corresponds to a 1d signal. """

import os

import numpy as np

import csv


def extract_create_dataset(raw_dataset_path, preprocess_dataset_path, object_list, time_steps, overlapping):

    file_counter = 0

    for folder in os.listdir(raw_dataset_path):

        print(folder)

        for file in os.listdir(os.path.join(raw_dataset_path, folder)):

            print(file)

            with open(os.path.join(raw_dataset_path, folder, file)) as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0

                matrix = []

                for row in csv_reader:

                    if line_count == 1:

                        matrix.append([float(i) for i in row])
                        line_count = line_count + 1

                    elif (line_count != 0) and ((line_count-1) <= time_steps):

                        matrix.append([float(i) for i in row])
                        line_count = line_count + 1

                    else:

                        line_count = line_count + 1

                    if (line_count != 0) and ((line_count-1) == time_steps):

                        for object in object_list:

                            if object in file:

                                name = object

                        # Create the pathname and create the folder if it does not exist
                        dataset_path = os.path.join(preprocess_dataset_path, folder)
                        if os.path.exists(dataset_path) is False:
                            os.mkdir(dataset_path)

                        dataset_path = os.path.join(dataset_path, name)
                        if os.path.exists(dataset_path) is False:
                            os.mkdir(dataset_path)

                        if len(np.array(matrix)) == time_steps:

                            with open(os.path.join(dataset_path, "data_" + str(file_counter) + ".csv"), 'w') as writeFile:
                                writer = csv.writer(writeFile)
                                writer.writerows(np.array(matrix).transpose())

                            writeFile.close()

                            file_counter = file_counter + 1

                            line_count = round(time_steps*(1-(overlapping/100)))

                            del matrix[0:line_count]

                            line_count = time_steps - line_count + 1

                        else:

                            line_count = 0
                            matrix = []



########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Define the sampling frequency of the data acquisition
sampling_frequency = 10

# Define the length of the vector (signal) corresponding to an instance
time_steps = sampling_frequency*2

# Define the percentage of overlapping
overlapping = 80

# List of objects held by the hand and gestures of the hand (Provided by the user)
object_list = ['Fourchette', 'Cuillere', 'Couteau', 'Assiette', 'Verre', 'Tasse', 'Casserole', 'Poele', 'Fouet',
              'Louche', 'PressoirePatates', 'Cafetiere', 'Rouleau', 'MainAVide']

# Define the path location of the raw dataset
raw_dataset_path = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\Signaux Total"

# Define the path location of the preprocess dataset
preprocess_dataset_path = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\CNNLSTM Dataset - V0"

########################################################################################################################
#                                                 Generate the Dataset                                                 #
########################################################################################################################

extract_create_dataset(raw_dataset_path, preprocess_dataset_path, object_list, time_steps, overlapping)
