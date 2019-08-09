import os
import time
from contextlib import redirect_stdout

import csv
import random
import numpy as np

import utilGlobalFunctions
import CNNLSTMModels


def load_data(path):

    """ This function loads the dataset given by its path location (input argument) """

    dataset_type = path.split(os.path.sep)[-1]

    if dataset_type == "Train":

        dataset_called = "training dataset"

    elif dataset_type == "Validation":

        dataset_called = "validation dataset"

    elif dataset_type == "Test":

        dataset_called = "testing dataset"

    else:

        print("Please verify the current folders in the main dataset directory")

    print("         The " + dataset_called + " are being loaded. Please wait ...")

    # Initialize the dataset
    dataset = []
    classes = []
    counter = -1

    # Load and transform (into the appropriate format) every data file in the training folder
    for data_class in os.listdir(path):

        counter = counter + 1
        classes.append(data_class)
        new_path = os.path.join(path, data_class)
        print("             The '" + data_class + "' class are being loaded ...")

        for data in os.listdir(new_path):

            data_path = os.path.join(new_path, data)

            if "DS_Store" not in data_path:
                with open(data_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    count_row = 0
                    for row in csv_reader:

                        if len(row) > 0:
                            vector = np.array([np.array([float(i)]) for i in list(row)])

                            if count_row == 0:

                                all_vector = vector
                                count_row = count_row + 1

                            else:

                                all_vector = np.concatenate((all_vector, vector), axis=1)
                                count_row = count_row + 1

                dataset.append([all_vector, data_class])

    random.seed()
    random.shuffle(dataset)

    # Transform the dataset in order to seperate the vector of data to the their corresponding class
    x = []
    y = []
    for i in dataset:
        x.append(i[0])
        y.append(i[1])

    print(np.array(x).shape)
    return np.array(x), y, classes


def print_model_summary(model, path):

    with open(path, 'w+') as f:
        print(model, file=f)

########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Name the Neural Network model (unique name)
model_name = "CNN LSTM - Model 1"

# Name the execution of this script
execution_name = "Test 1"

# Give the main path of the dataset to exploit (dataset location)
dataset_path = "C:\\Users\\Julien\\Documents\\Julien\\Université\\Recherche\\Supervision\\2018 - Clément Rendu\\Signaux\\CNNLSTM Dataset - V1"

# Define the number of subsquence
subsequence = 4

# Parameters for the training
# A set of N samples. The samples in a batch are processed independently, in parallel.
# If training, a batch results is only one update to the model.
batch_size = 128

# The number of iterations (iteration = one pass over the entire dataset)
epochs = 200

optimizer = "adam"
# The choices are :
#                   "sgd" - Stochastic gradient descent optimizer.
#                   "rmsprop" - This optimizer is usually a good choice for recurrent neural networks.
#                   "adagrad" - Adagrad is an optimizer with parameter-specific learning rates, which are adapted
#                               relative to how frequently a parameter gets updated during training. The more
#                               updates a parameter receives, the smaller the learning rate.
#                   "adadelta" - Adadelta is a more robust extension of Adagrad that adapts learning rates based on a
#                                moving window of gradient updates, instead of accumulating all past gradients. This
#                                way, Adadelta continues learning even when many updates have been done.
#                   "adam"
#                   "adamax" - It is a variant of Adam based on the infinity norm.
#                   "nadam" - Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with
#                             Nesterov momentum.

loss = 'categorical_crossentropy'
# The choices are :
#                   "mean_squared_error"
#                   "mean_absolute_error"
#                   "mean_absolute_percentage_error"
#                   "mean_squared_logarithmic_error"
#                   "squared_hinge"
#                   "hinge"
#                   "categorical_hinge"
#                   "logcosh"
#                   "categorical_crossentropy"
#                   "sparse_categorical_crossentropy"
#                   "binary_crossentropy"
#                   "kullback_leibler_divergence"
#                   "poisson"
#                   "cosine_proximity"

metrics = ["accuracy"]
# The choices are :
#                   "binary_accuracy"
#                   "categorical_accuracy"
#                   "sparse_categorical_accuracy"
#                   "top_k_categorical_accuracy"
#                   "sparse_top_k_categorical_accuracy"

########################################################################################################################
#                                             Create Pathnames and Folders                                             #
########################################################################################################################

print("\n\nWe are creating different folders for the execution of this script. Please wait ...")

# Get the working pathname of the deep learning toolbox
work_path = os.getcwd()

# Get the head pathname
head_path = os.path.split(work_path)[0]
# Create the pathname of the "Results" folder
result_path = os.path.join(head_path, "Results")
# Create the "Results" folder if it does not exist
if os.path.exists(result_path) is False:
    os.mkdir(result_path)

# Create the pathname of the "Model n" folder
model_path = os.path.join(result_path, model_name)
# Create the "Model n" folder if it does not exist
if os.path.exists(model_path) is False:
    os.mkdir(model_path)

# Get the current date and time
current_date = time.strftime("%d-%m-%Y")
current_time = time.strftime("%Hh%Mmin%Ssec")
# Create the folder name of this execution
new_execution_name = execution_name + " - " + current_date + " - " + current_time
# Create the pathname of the "Test n - Date - Time" folder
test_path = os.path.join(model_path, new_execution_name)
# Create the "Test n - Date - Time" folder if it does not exist
if os.path.exists(test_path) is False:
    os.mkdir(test_path)

# Open a .txt file that contains all information about the current execution
f = open(os.path.join(test_path, 'Information.txt'), 'w+')
print("#######################################################################################################", file=f)
print("#                                           User Parameters                                           #", file=f)
print("#######################################################################################################", file=f)

print("\nThe deep neural network is a CNN LSTM.", file=f)
print("\n\nThe dataset is : " + dataset_path, file=f)
print("\n\nBatch size : %s" % batch_size, file=f)
print("Epoch number : %s" % epochs, file=f)
print("Optimizer type : %s" % optimizer, file=f)
print("Loss type : %s" % loss, file=f)

########################################################################################################################
#                                                    Load the Dataset                                                  #
########################################################################################################################

print("#######################################################################################################", file=f)
print("#                                              Dataset                                                #", file=f)
print("#######################################################################################################", file=f)

# Create the different paths where the dataset is located
training_path = os.path.join(dataset_path, "Train")
validation_path = os.path.join(dataset_path, "Validation")
testing_path = os.path.join(dataset_path, "Test")

print("We are loading the training, validation and testing datasets. Please wait ...")

print("     The script is loading the training dataset. Please wait ...")

# Get the current clock to measure the time consuming for loading the training dataset
start = time.clock()
# Load the training dataset
x_train, y_train, train_classes = load_data(training_path)
# Print the time consuming for loading the training dataset
totalTime = time.clock() - start

print("\nThe total time for loading the training dataset is : %f second(s)" % totalTime, file=f)
print("\n           The total time for loading the training dataset is : %f second(s)" % totalTime)


print("\n\n     The script is loading the validating dataset. Please wait ...")

# Get the current clock to measure the time consuming for loading the validating dataset
start = time.clock()
# Load the validating dataset
x_validation, y_validation, validation_classes = load_data(validation_path)
# Print the time consuming for loading the validating dataset
totalTime = time.clock() - start

print("The total time for loading the validating dataset is : %f second(s)" % totalTime, file=f)
print("\n           The total time for loading the validating dataset is : %f second(s)" % totalTime)


print("\n\n     The script is loading the testing dataset. Please wait ...")

# Get the current clock to measure the time consuming for loading the testing dataset
start = time.clock()
# Load the testing dataset
x_test, y_test, test_classes = load_data(testing_path)
# Print the time consuming for loading the testing dataset
totalTime = time.clock() - start

print("The total time for loading the testing dataset is : %f second(s)" % totalTime, file=f)
print("\n           The total time for loading the testing dataset is : %f second(s)" % totalTime)

########################################################################################################################
#                                  Get information and Transform Labels of the Dataset                                 #
########################################################################################################################

print("\n\nWe are computing the number of instances for each part of the dataset.")

# Compute the number of instances of the training dataset
training_instance_number = len(x_train)
# Compute the number of instances of the validating dataset
validating_instance_number = len(x_validation)
# Compute the number of instances of the testing dataset
testing_instance_number = len(x_test)

print("     There are %s instances in the training dataset." % training_instance_number)
print("\nThere are %s instances in the training dataset." % training_instance_number, file=f)

print("     There are %s instances in the validating dataset." % validating_instance_number)
print("There are %s instances in the validating dataset." % validating_instance_number, file=f)

print("     There are %s instances in the testing dataset." % testing_instance_number)
print("There are %s instances in the testing dataset." % testing_instance_number, file=f)

print("\n\nWe are transforming the data shape into subsquence. Please wait ...")
n_steps = subsequence
modulo = x_train.shape[1] % n_steps

if modulo == 0:

    n_length = int(x_train.shape[1] / n_steps)

else:

    while x_train.shape[1] % n_steps != 0:

        n_steps = n_steps + 1

    n_length = int(x_train.shape[1] / n_steps)

x_train = x_train.reshape((x_train.shape[0], n_steps, n_length, x_train.shape[2]))
x_validation = x_validation.reshape((x_validation.shape[0], n_steps, n_length, x_validation.shape[2]))
x_test = x_test.reshape((x_test.shape[0], n_steps, n_length, x_test.shape[2]))

print("\n\nWe are transforming label of the dataset into one hot. Please wait ...")

train_encoder, train_encoded_string_label, y_train_one_hot = utilGlobalFunctions.encodeStringLabel(y_train)
validation_encoder, validation_encoded_string_label, y_validation_one_hot = utilGlobalFunctions.encodeStringLabel(y_validation)
test_encoder, test_encoded_string_label, y_test_one_hot = utilGlobalFunctions.encodeStringLabel(y_test)

########################################################################################################################
#                                           Build the Residual Network Model                                           #
########################################################################################################################

print("#######################################################################################################", file=f)
print("#                                   Build the Residual Network Model                                  #", file=f)
print("#######################################################################################################", file=f)

print("\n\nWe are building the CNN LSTM model. Please wait ...")

# Get the current clock to measure the time consuming for building the model
start = time.clock()
# Build the model
model = CNNLSTMModels.CNNLSTMModelV1(input_shape=np.shape(x_train[0]), classes=len(train_classes))  # A changer pour choisir le modele voulu
# Print the time consuming for building the model
totalTime = time.clock() - start

print("     The total time for building the ResNet model is : %f second(s)" % totalTime)
print("\nThe total time for building the ResNet model is : %f second(s)" % totalTime, file=f)

# Print a summary of the model
print(model.summary())
with redirect_stdout(f):
    model.summary()

########################################################################################################################
#                                                   Train the Model                                                    #
########################################################################################################################

print("#######################################################################################################", file=f)
print("#                                          Train the Model                                            #", file=f)
print("#######################################################################################################", file=f)

# Define the compiler - peut etre a changer pour definir avec plus de precision l'optimizer (definir le learning rate)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print('\n\nThe model is training ...')

# Get the current clock to measure the time consuming for training stage of the model
start = time.clock()
# Train the model
model_history = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                          validation_data=(x_validation, y_validation_one_hot))
# Print the time consuming for training stage of the model
totalTime = time.clock() - start

print("     The total time for training the ResNet model is : %f second(s)" % totalTime)
print("\nThe total time for training the ResNet model is : %f second(s)" % totalTime, file=f)

# Plot the accuracy and loss values obtained during the training
utilGlobalFunctions.plotKNNPerformances(model_history)

########################################################################################################################
#                                                  Evaluate the Model                                                  #
########################################################################################################################

print("#######################################################################################################", file=f)
print("#                                          Evaluate the Model                                         #", file=f)
print("#######################################################################################################", file=f)

""" Returns the loss value & metrics values for the model in test mode """

print('\n\nThe model is evaluating ...')

# Get the current clock to measure the time consuming for evaluating the model
start = time.clock()
# Evaluate the model with the testing dataset
keras_score = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)
# Print the time consuming for evaluating the model
totalTime = time.clock() - start

print("     The total time for evaluating the ResNet model is : %f second(s)" % totalTime)
print("\nThe total time for evaluating the ResNet model is : %f second(s)" % totalTime, file=f)

print("\n     Keras Performances of the Trained Model\n")
print("\n     Keras Performances of the Trained Model\n", file=f)
print("         Test loss : ", keras_score[0])
print("         Test loss : ", keras_score[0], file=f)
print("         Test accuracy : ", keras_score[1])
print("         Test accuracy : ", keras_score[1], file=f)

########################################################################################################################
#                                            Predict and Evaluate the Model                                            #
########################################################################################################################

print("#######################################################################################################", file=f)
print("#                                   Predict and Evaluate the Model                                    #", file=f)
print("#######################################################################################################", file=f)

""" We predict the class for each instance of x_test in order to compute classical performance metrics """

print('\n\nThe model is predicting ...')

# Get the current clock to measure the time consuming for predicting the testing instances with this model
start = time.clock()
# Predict the class for each instance of x_test
y_predicted = model.predict(x_test)
# Print the time consuming for predicting the testing instances
totalTime = time.clock() - start

print("     The total time for predicting the testing instances is : %f second(s)" % totalTime)
print("\nThe total time for predicting the testing instances is : %f second(s)" % totalTime, file=f)


# Convert the classification for each instance of x_test
y_predicted, string_predicted_label = utilGlobalFunctions.decodeOneHotLabel(test_encoder, y_predicted)

# Compute the performances of the model
accuracy, balanced_accuracy, report, cohen_kappa = \
    utilGlobalFunctions.performanceMetrics(y_test, string_predicted_label, list(test_encoder.classes_))

# Display in the performances in the console
print("     Classical Performances of the Trained Model\n")
print("     Classical Performances of the Trained Model\n", file=f)
print("         Accuracy : ", accuracy)
print("         Accuracy : ", accuracy, file=f)
print("         Balanced_accuracy : ", balanced_accuracy)
print("         Balanced_accuracy : ", balanced_accuracy, file=f)
print("         Cohen's Kappa : ", cohen_kappa)
print("         Cohen's Kappa : ", cohen_kappa, file=f)
print("         Report : \n", report)
print("         Report : \n", report, file=f)

print("We are saving the model into the corresponding 'Result' folder. ")

########################################################################################################################
#                                          Evaluate the Model to Get the Top N                                         #
########################################################################################################################

print("#######################################################################################################", file=f)
print("#                                  Evaluate the Model to Get the Top N                                #", file=f)
print("#######################################################################################################", file=f)

print('\n\nThe model is predicting to compute the Top N ...')

# Get the current clock to measure the time consuming for predicting the testing instances with this model
start = time.clock()
# Predict the class for each instance of x_test
y_predicted_prob = model.predict_proba(x_test)
# Print the time consuming for predicting the testing instances
totalTime = time.clock() - start

print("     The total time for predicting the testing instances is : %f second(s)" % totalTime)
print("\nThe total time for predicting the testing instances is : %f second(s)" % totalTime, file=f)

# Sort the probabilities results in descending order
scores = (-y_predicted_prob).argsort()

for top_n in range(len(test_classes)):

    new_score = []
    counter = 0

    for instance in scores:

        if np.isin(test_encoded_string_label[counter], instance[0:top_n+1]):

            new_score.append(test_encoded_string_label[counter])

        else:

            new_score.append(instance[0])

        counter = counter + 1

    # Compute the performances of the model
    accuracy, balanced_accuracy, report, cohen_kappa = \
        utilGlobalFunctions.performanceMetrics(test_encoded_string_label, new_score, list(test_encoder.classes_))

    print("\n\nTop " + str(top_n+1) + " Results")
    print("\nTop " + str(top_n+1) + " Results", file=f)

    # Display in the performances in the console
    print("\n     Classical Performances of the Trained Model\n")
    print("     Classical Performances of the Trained Model\n", file=f)
    print("         Accuracy : ", accuracy)
    print("         Accuracy : ", accuracy, file=f)
    print("         Balanced_accuracy : ", balanced_accuracy)
    print("         Balanced_accuracy : ", balanced_accuracy, file=f)
    print("         Cohen's Kappa : ", cohen_kappa)
    print("         Cohen's Kappa : ", cohen_kappa, file=f)
    print("         Report : \n", report)
    print("         Report : \n", report, file=f)


print("We are saving the model into the corresponding 'Result' folder. ")

# Save the model
# model.save(os.path.join(test_path, "ResNetModel.h5"))

# Close the .txt file
f.close()


