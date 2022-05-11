import math
from model2 import *
from preprocess import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

def online():
    train, tr_labels, test, test_labels, size = get_data(False)
    model = Sequential()
    size = 16
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    # model.add(Dense(18,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse')
    model.fit(x=train,y=tr_labels,
          validation_data=(test,test_labels),
          batch_size=128,epochs=400)
    
    model.summary()
    # loss_df = pd.DataFrame(model.history.history)
    # loss_df.plot(figsize=(12,8))

    # plt.show()

    y_pred = model.predict(test)
    from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(test_labels, y_pred))  
    print('MSE:', metrics.mean_squared_error(test_labels, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
    print('VarScore:',metrics.explained_variance_score(test_labels,y_pred))

def main():
    
    print("main")
    train, tr_labels, test, test_labels, size = get_data(False)
    
    # model = model_boi()

    model = Sequential()
    # size = 16
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(size,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse')
    # model.compile(model.optimizer, model.loss_func)
    model.fit(x=train,y=tr_labels,
          validation_data=(test,test_labels),
          batch_size=128,epochs=400)
    
    print(model.summary())
    # loss_df = pd.DataFrame(model.history.history)
    # loss_df.plot(figsize=(12,8))

    # plt.show()

    y_pred = model.predict(test)
    
    print('MAE:', metrics.mean_absolute_error(test_labels, y_pred))  
    print('MSE:', metrics.mean_squared_error(test_labels, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
    print('VarScore:',metrics.explained_variance_score(test_labels,y_pred))


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

        if batch % 100 == 0:
            print(loss)

def test(model, test_inputs, test_labels):
    last_batch_index = 0
    total_batches = math.ceil(test_inputs.shape[0]/model.batch_size)
    print(total_batches)
    accuracies = []
    for b in range(total_batches):
        if(last_batch_index>= (test_inputs.shape[0]- model.batch_size)):
            batch_inputs = test_inputs[last_batch_index:]
            batch_labels = test_labels[last_batch_index:]
        else:
            batch_inputs = test_inputs[last_batch_index:last_batch_index+model.batch_size]
            batch_labels = test_labels[last_batch_index:last_batch_index+model.batch_size]
         
        logits = model.call(batch_inputs, True) # get prediction
        acc = model.accuracy(logits, batch_labels)
        accuracies.append(acc)

        if(b % 10 == 0):

            print("test accuracy at batch", b, ":", acc)

        last_batch_index += model.batch_size
        
    avg_acc = sum(accuracies)/len(accuracies)
    return avg_acc


def main2():
    train_input, tr_labels, test_input, test_labels, size = get_data(False)

    model = lin_reg()
    total_epochs = 100
    for e in range(total_epochs):
        train(model, train_input, tr_labels)
    
    acc = test(model, test_input, test_labels)
    y_pred = model.predict(test_input)
    from sklearn import metrics
    print('MAE:', metrics.mean_absolute_error(test_labels, y_pred))  
    # print('MSE:', metrics.mean_squared_error(test_labels, y_pred))  
    print('RMSE:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
    print('VarScore:',metrics.explained_variance_score(test_labels,y_pred))

    print("RMSE:", np.sqrt(acc))
    
main()
# main2()