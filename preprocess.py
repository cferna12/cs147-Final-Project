import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

def classify_price(x):
    range1 = 400000
    range2 = 800000
    range3 = 1200000
    
    if x < range1:
        return 0
        return range2
        return '400,000'
    elif x < range2:
        return 1
        return range1
        return '800,000'
    elif x < range3:
        return 2
        return range3
        return '1,200,000'
    else:
        return 3
        # return 2000000
        return '>1200000'


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

# data = remove_outliers_iqr(data, 'price')

#maybe only remove outliers for float not caregories
# for column in data.columns:
#     data = remove_outliers_iqr(data, column)

# data['price'] = data['price'].apply(classify_price)

def bar_plot(data, col, xlabel):
    fig, my_ax = plt.subplots()
    hist = data[col].value_counts().plot(kind='bar')
    my_ax.set_xlabel(xlabel)
    plt.xticks(rotation = 0)
    plt.show(hist)

def get_data():
    file_path = "kc_house_data.csv"
    data = pd.read_csv(file_path)

    data['yr_renovated'] = np.where(data['yr_renovated'] > 0, 1, 0)
    data['age'] = data['yr_built'].apply(lambda x: 2015-x)

    data['price'] = data['price'].apply(classify_price)
    data = data.drop(columns=['id', 'date', 'yr_built']) 

    msk = np.random.rand(len(data)) < 0.8

    train = data[msk]
    test = data[~msk]
    
    train_labels = train['price']
    test_labels = test['price']

    # train_labels = train['price'].apply(classify_price)
    # test_labels = test['price'].apply(classify_price)

    train = train.drop(columns=['price']) 
    test = test.drop(columns='price')
    # labels.to_numpy()

    return  train, train_labels, test, test_labels


# bar_plot('price', xlabel = 'House price')
# bar_plot('bedrooms', '# bedrooms')
# bar_plot('condition', 'condition')

# print(data.columns)
# d = data[['price', 'grade', 'condition']]
# print(d['grade'])

# corr = d.corr(method='pearson')
# print(corr)


