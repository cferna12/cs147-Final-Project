import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
import seaborn as sns

def classify_price(x):
    '''
    Mapping function that converts prices into sepearate categories 
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


def outliers_range_iqr(data, column, distance = 1.5):
    # Finds the iqr range of a column
    iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
    lower_threshold = data[column].quantile(0.25) - (iqr * distance)
    upper_threshold = data[column].quantile(0.75) + (iqr * distance)
    return lower_threshold, upper_threshold

def find_outliers_iqr(data, column):
    # Identifiest the outliers in a column
    lower_threshold, upper_threshold = outliers_range_iqr(data, column)
    outliers = []
    for i in data[column]:
        # print(i)
        if i > upper_threshold:
            outliers.append(i)
        elif i < lower_threshold:
            outliers.append(i)
        else: 
            pass
    return outliers

def remove_outliers_iqr(data, column):
    # Removes the outliers in a column
    outliers = find_outliers_iqr(data, column)
    outliers = pd.Series(outliers)
    data_new = data[~data[column].isin(outliers)]
    return data_new



def apply_basement(x):
    '''
    Mapping function that converts sqft_basement to binary (basement/no basement)
    '''
    if x>0:
        return 1
    else:
        return 0


def bar_plot(data, col, xlabel):
    '''
    Creates a bar plot based on a column of interest
    '''
    fig, my_ax = plt.subplots()
    hist = data[col].value_counts().plot(kind='bar')
    my_ax.set_xlabel(xlabel)
    plt.xticks(rotation = 0)
    plt.show(hist)


def get_data(is_classification):
    '''
    Function that returns the training and testing data as numpy arrays
    '''
    file_path = "kc_house_data.csv"
    data = pd.read_csv(file_path)

    data['yr_renovated'] = np.where(data['yr_renovated'] > 0, 1, 0)
    data['age'] = data['yr_built'].apply(lambda x: 2015-x)
    data['sqft_basement'] = data['sqft_basement'].apply(apply_basement)

    data = data.drop(columns=['id', 'date', 'yr_built','lat', 'long'])

    # top_feats = find_corr(data)
    # print(top_feats)
    # data = data[top_feats]
    # data = remove_outliers_iqr(data, 'price')
    # for col in data.columns:
    #     if col != 'price':
    #         new_data = data[[col, 'price']]
    #         print(new_data)
    #         new_data = remove_outliers_iqr(new_data, 'price')
    #         data[col] = new_data[col]

    # data = data.dropna()

    if(is_classification):
        data['price'] = data['price'].apply(classify_price)
    
    print(data)
    x = data.drop('price',axis =1).values
    y = data['price'].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    mask = np.random.rand(len(data)) < 0.75
    train = data[mask]
    test = data[~mask]
    
    train_labels = train['price']
    test_labels = test['price']

    return X_train, y_train, X_test,  y_test, X_train.shape[1]
    
    return  train.to_numpy(), train_labels.to_numpy(), test.to_numpy(), test_labels.to_numpy(), len(train.columns)



def find_corr(train):
    '''
    Helper function used to find correlation with price
    '''
    top_features = train.corr()[['price']].sort_values(by=['price'],ascending=False).head(10)
    plt.figure(figsize=(5,10))
    sns.heatmap(top_features,cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=-1)
    # plt.show()
    print(top_features.index)
    return top_features.index


# get_data(True)

