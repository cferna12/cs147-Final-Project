import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def classify_price(x):
    '''
    Mapping function that converts prices into sepearate categories based on which range
    they fall in

    :param x: numerical price values
    :returns: categorical price value
    '''
    range1 = 400000
    range2 = 800000
    range3 = 1200000
    
    if x < range1:
        return 0
    elif x < range2:
        return 1
    elif x < range3:
        return 2
    else:
        return 3


def apply_basement(x):
    '''
    Mapping function that converts sqft_basement to binary (basement/no basement)
    '''
    if x>0:
        return 1
    else:
        return 0


def histogram(data, col, xlabel):
    '''
    Creates a histogram based on a column of interest

    :param data: pandas dataset being used
    :param col: column for which constructing histogram
    :param xlabel: label for the x-axis
    '''
    fig, my_ax = plt.subplots()
    hist = data[col].value_counts().plot(kind='bar')
    my_ax.set_xlabel(xlabel)
    plt.xticks(rotation = 0)
    plt.show(hist)


def get_data(is_classification):
    '''
    Function that returns the training and testing data as numpy arrays

    :param is_classification: bool representing if using classification model

    :return: training and testing inputs and labels, and total columns (needed for linear regression model)
    '''
    file_path = "kc_house_data.csv"
    data = pd.read_csv(file_path)

    #apply map functions to change necessary columns
    data['yr_renovated'] = np.where(data['yr_renovated'] > 0, 1, 0)
    data['age'] = data['yr_built'].apply(lambda x: 2015-x)
    data['sqft_basement'] = data['sqft_basement'].apply(apply_basement) 

    data = data.drop(columns=['id', 'date', 'yr_built','lat', 'long']) #drop columns

    if(is_classification): #if classification, apply price map
        data['price'] = data['price'].apply(classify_price)
    
    inputs = data.drop('price',axis =1).values
    labels = data['price'].values
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size = 0.2)

    #scale inputs
    train_inputs = RobustScaler().fit_transform(train_inputs.astype(np.float))
    test_inputs = RobustScaler().fit_transform(test_inputs.astype(np.float))

    return train_inputs, train_labels, test_inputs, test_labels, train_inputs.shape[1]



