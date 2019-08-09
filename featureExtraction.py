import os
import csv

import numpy as np
import pandas as pd
from scipy import stats
from librosa import feature
import math

""" The data input is organized in column. In other terms, each sensor has a serie of values. Each column corresponding
    to one sensor and in this column each line correspond to a sample of the corresponding sensor. """


def extract_min(data):

    min_coefficients = np.min(data, axis=0)
    # print(min_coefficients)

    return min_coefficients


def extract_max(data):

    max_coefficients = np.max(data, axis=0)
    # print(max_coefficients)

    return max_coefficients


def extract_average(data):

    average = np.mean(data, axis=0)
    # print(average)

    return average


def extract_var_std(data):

    standard_deviation = np.std(data, axis=0)
    variance = standard_deviation*standard_deviation
    # print(standard_deviation)
    # print(variance)

    return variance, standard_deviation


def extract_skewness(data):

    skewness = stats.skew(data, axis=0)
    # print(skewness)

    return skewness


def extract_kurtosis(data):

    kurtosis = stats.kurtosis(data, axis=0)
    # print(kurtosis)

    return kurtosis


def extract_mean_absolute_deviation(data):

    df = pd.DataFrame(data)
    mad_coefficients = df.mad(axis=0)
    # print(np.array(mad_coefficients))

    return np.array(mad_coefficients)


def extract_waveform_length(data):

    df = pd.DataFrame(data)
    waveform_length_coefficients = []

    for i in df:

        my_array = np.array(df[i])
        waveform_length = sum(abs(my_array[1:] - my_array[:-1]))
        waveform_length_coefficients.append(waveform_length)

    # print(np.array(waveform_length_coefficients))

    return np.array(waveform_length_coefficients)


def extract_zero_crossing_rate(data):

    df = pd.DataFrame(data)
    zero_crossing_rate = []

    for i in df:

        my_array = np.array(df[i])
        zero_crossing_rate.append(((my_array[:-1] * my_array[1:]) < 0).sum())

    # print(np.array(zero_crossing_rate))
    return np.array(zero_crossing_rate)


def extract_mean_crossing_rate(data):

    df = pd.DataFrame(data)
    mean_crossing_rate = []

    for i in df:

        my_array = np.array(df[i])

        mean = np.mean(my_array)
        counter = 0
        above = 0
        under = 0

        for j in range(len(my_array)):

            if (my_array[j] > mean) and (above == 0):

                counter = counter + 1
                above = 1
                under = 0

            elif (my_array[j] < mean) and (under == 0):

                counter = counter + 1
                above = 0
                under = 1

        mean_crossing_rate.append(counter-1)

    # print(np.array(zero_crossing_rate))
    return np.array(mean_crossing_rate)


def extract_energies(data):

    df = pd.DataFrame(data)
    energy_coefficients = []

    for i in df:

        my_array = np.array(df[i])
        energy = sum(abs(my_array)**2)
        energy_coefficients.append(energy)

    # print(np.array(energy_coefficients))
    return np.array(energy_coefficients)


def extract_cross_correlation(data):

    df = pd.DataFrame(data)
    correlations = []

    for i in df:

        for j in df:

            if i <= j:

                correlation_coefficient = np.correlate(np.array(df[i]), np.array(df[j]))

                correlations.append(float(correlation_coefficient))

    # print(np.array(correlations))
    return np.array(correlations)



def feature_extraction(data):

    tmp = extract_min(data)
    tmp = np.append(tmp, extract_max(data))
    tmp = np.append(tmp, extract_average(data))
    tmp = np.append(tmp, extract_var_std(data))
    tmp = np.append(tmp, extract_skewness(data))
    tmp = np.append(tmp, extract_kurtosis(data))
    tmp = np.append(tmp, extract_mean_absolute_deviation(data))
    tmp = np.append(tmp, extract_waveform_length(data))
    tmp = np.append(tmp, extract_mean_crossing_rate(data))
    tmp = np.append(tmp, extract_energies(data))
    tmp = np.append(tmp, extract_cross_correlation(data))

    return tmp




