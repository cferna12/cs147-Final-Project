
import math
import numpy as np
from preprocess import *
from model import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def train(model, train_inputs, train_labels):
    print("train")

    # remainder =  train_inputs.shape[0] % model.batch_size
    # print(remainder)
    # train_inputs = train_inputs[:-remainder]
    # train_labels = labels[:-remainder]
    total_batches = math.ceil(train_inputs.shape[0]/model.batch_size)
    # print(total_batches)

    # print("t batfches", total_batches)

    indices = range(train_inputs.shape[0]) #create indices 
    shuffled_indices = tf.random.shuffle(indices) #shuffle indices
    train_inputs = tf.gather(train_inputs, shuffled_indices) #shuffle inputs
    train_labels = tf.gather(train_labels, shuffled_indices) #shuffle labels
    last_batch_index = 0
    for batch in range(total_batches):
       # print(b, last_batch_index)
        if(last_batch_index>= (train_inputs.shape[0]- model.batch_size)):
            batch_inputs = train_inputs[last_batch_index:]
            batch_labels = train_labels[last_batch_index:]
           # print(batch_inputs.shape)
        else:
            batch_inputs = train_inputs[last_batch_index:last_batch_index+model.batch_size]
            batch_labels = train_labels[last_batch_index:last_batch_index+model.batch_size]

        with tf.GradientTape() as tape:
            probs = model.call(batch_inputs) # this calls the call function conveniently
            loss = model.loss(probs, batch_labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if batch % 500 == 0:
            print(loss)


def test(model, test_inputs, test_labels):
    print("testing")
    last_batch_index = 0
    total_batches = math.ceil(test_inputs.shape[0]/model.batch_size)
    print("total_batches", total_batches)
    accuracies = []
    for b in range(total_batches):
        if(last_batch_index>= (test_inputs.shape[0]- model.batch_size)):
            batch_inputs = test_inputs[last_batch_index:]
            batch_labels = test_labels[last_batch_index:]
        else:
            batch_inputs = test_inputs[last_batch_index:last_batch_index+model.batch_size]
            batch_labels = test_labels[last_batch_index:last_batch_index+model.batch_size]
         
        logits = model.call(batch_inputs) # get prediction
        acc = model.accuracy2(logits, batch_labels)
        accuracies.append(acc)

        if(b % 10 == 0):

            print("test accuracy at batch", b, ":", acc)

        last_batch_index += model.batch_size
        
    avg_acc = sum(accuracies)/len(accuracies)
    print(avg_acc)
    return avg_acc


def main():
    train_inputs, train_labels, test_inputs, test_labels, columns = get_data(True)
    print(train_inputs.shape)

    num_classes = 4
    train_labels =tf.one_hot(train_labels, num_classes)
    test_labels = tf.one_hot(test_labels, num_classes)
    
    model = model_boi(num_classes)

    # total_epochs = 20
    # for e in range(total_epochs):
    #     print("E: ", e)
    #     train(model, train_inputs, train_labels)


    # acc = test(model, test_inputs, test_labels)
    
    model = Sequential()
    model.add(Dense(200,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(num_classes))
    model.compile(optimizer='Adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits= True),
    metrics=tf.keras.metrics.CategoricalCrossentropy(from_logits = True))


    model.fit(x=train_inputs,y=train_labels,
          validation_data=(test_inputs,test_labels),
          batch_size=128,epochs=100)
    model.summary()
    model.evaluate(test_inputs, test_labels)



main()