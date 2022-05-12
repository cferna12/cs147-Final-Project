import tensorflow as tf
import math
import numpy as np
from preprocess import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

    
def main():
    '''
    Builds, trains, and tests classification model.
    Creates a plot of the loss history 
    '''
    train_inputs, train_labels, test_inputs, test_labels, columns = get_data(True)
    num_classes = 4 #predicting 4 house ranges

    train_labels =tf.one_hot(train_labels, num_classes) #one hot encode labels
    test_labels = tf.one_hot(test_labels, num_classes) #one hot encode labels
    
    model = Sequential()
    model.add(Dense(500,activation='relu'))
    model.add(Dense(300,activation='relu'))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes))
    model.compile(optimizer='Adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits= True),
    metrics=tf.keras.metrics.CategoricalAccuracy())

    total_epochs = 30
    model.fit(x=train_inputs,y=train_labels,
          validation_data=(test_inputs,test_labels),
          batch_size=120,epochs=total_epochs)
    model.summary()

    history = pd.DataFrame(model.history.history) #get model history for plot
    
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.legend()
    plt.title("Loss history over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    model.evaluate(test_inputs, test_labels)
    

if __name__ == '__main__':
    main()