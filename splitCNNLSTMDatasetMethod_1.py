import os

import random

import shutil


def split_dataset(preprocess_dataset_path, splitting_dataset_path, object_list, train_ratio, validation_ratio,
                  test_ratio):

    for object in object_list:

        new_file_path_list = []

        for folder in os.listdir(preprocess_dataset_path):

            for sub_folder in os.listdir(os.path.join(preprocess_dataset_path, folder)):

                new_preprocess_dataset_path = os.path.join(preprocess_dataset_path, folder)

                if object == sub_folder:

                    file_list = os.listdir(os.path.join(new_preprocess_dataset_path, sub_folder))

                    for file_name in file_list:

                        new_file_path_list.append(os.path.join(new_preprocess_dataset_path, sub_folder, file_name))

        random.seed()
        random.shuffle(new_file_path_list)

        file_number = len(new_file_path_list)

        train_file_indices = round(train_ratio * file_number)
        validation_file_indices = train_file_indices + round(validation_ratio * file_number)

        counter = 0

        for file_path in new_file_path_list:

            file = os.path.split(file_path)

            if counter < train_file_indices:

                if os.path.exists(os.path.join(splitting_dataset_path, "Train", object)) is False:
                    os.mkdir(os.path.join(splitting_dataset_path, "Train", object))

                shutil.move(file_path, os.path.join(splitting_dataset_path, "Train", object, file[1]))
                counter = counter + 1

            elif counter < validation_file_indices:

                if os.path.exists(os.path.join(splitting_dataset_path, "Validation", object)) is False:
                    os.mkdir(os.path.join(splitting_dataset_path, "Validation", object))

                shutil.move(file_path,
                            os.path.join(splitting_dataset_path, "Validation", object, file[1]))
                counter = counter + 1

            else:

                if os.path.exists(os.path.join(splitting_dataset_path, "Test", object)) is False:
                    os.mkdir(os.path.join(splitting_dataset_path, "Test", object))

                shutil.move(file_path,
                            os.path.join(splitting_dataset_path, "Test", object, file[1]))
                counter = counter + 1


########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Define the path location of the preprocess dataset
preprocess_dataset_path = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\CNNLSTM Dataset - V0"

# Define the path location of the splitting dataset
splitting_dataset_path = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\CNNLSTM Dataset - V1"

# List of objects held by the hand and gestures of the hand (Provided by the user)
object_list = ['Fourchette', 'Cuillere', 'Couteau', 'Assiette', 'Verre', 'Tasse', 'Casserole', 'Poele', 'Fouet',
              'Louche', 'PressoirePatates', 'Cafetiere', 'Rouleau', 'MainAVide']

# Percentage of instances being the training dataset
train_ratio = 0.7

# Percentage of instances being the validating dataset
validation_ratio = 0.15

# Percentage of instances being the testing dataset
test_ratio = 0.15

########################################################################################################################
#                                                 Generate the Dataset                                                 #
########################################################################################################################

split_dataset(preprocess_dataset_path, splitting_dataset_path, object_list, train_ratio, validation_ratio, test_ratio)