from preprocess import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import tensorflow as tf

def main():
    '''
    Builds, trains, and tests linear regressoin model.
    Creates a scatter plot of actual house vs predicted house price
    '''
    train, tr_labels, test, test_labels, size = get_data(False)

    model = Sequential() 
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(1))

    compile_metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()] #use rmse and mae as metrics
    model.compile(optimizer='Adam',loss='mse', metrics = compile_metrics)

    model.fit(x=train,y=tr_labels,
          validation_data=(test,test_labels),
          batch_size=120,epochs=500)
    
    predicted_labels = model.predict(test) #predict house price using our trained linear regression model
    r2 = metrics.r2_score(test_labels, predicted_labels)
    var = metrics.explained_variance_score(test_labels,predicted_labels)
    print("R2:" , r2)
    print('VarScore:', var)
    model.evaluate(test, test_labels) 

    #create scatter plot 
    plt.scatter(test_labels, predicted_labels)
    plt.plot(test_labels,test_labels,'r')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted Price")
    plt.show()

if __name__ == '__main__':
    main()