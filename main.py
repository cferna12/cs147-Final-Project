
import numpy as np
from preprocess import *
from model import *

def train(model, data, labels):
    print("train")
    total_batches = data.shape[0]//model.batch_size
    print(total_batches)

    indices = range(data.shape[0]) #create indices 
    shuffled_indices = tf.random.shuffle(indices) #shuffle indices
    data = tf.gather(data, shuffled_indices) #shuffle inputs
    labels = tf.gather(labels, shuffled_indices) #shuffle labels

    for b in range(total_batches):
        curr_idx = model.batch_size*b
        batch = data[model.batch_size*b:model.batch_size*(b+1)]
        batch_labels = labels[model.batch_size*b:model.batch_size*(b+1)]
        # print('batch labels shape', batch_labels.shape)

        with tf.GradientTape() as tape:
            probs = model.call(batch) # this calls the call function conveniently
            loss = model.loss(probs, batch_labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if b % 500 == 0:
            print(loss)


def test(model, inputs, labels):
    total_batches = inputs.shape[0]//model.batch_size

    total_acc = 0
    print(total_batches)
    for b in range(total_batches):
        batch = inputs[model.batch_size*b:model.batch_size*(b+1)]
        batch_labels = labels[model.batch_size*b:model.batch_size*(b+1)]
        probs = model.call(batch)
        # loss = model.loss(probs, labels)
        acc = model.accuracy(probs, batch_labels)

        if b %50 == 0:
            print(acc)
        total_acc+= acc
    
    accuracy = total_acc/total_batches
    print("accuracy:", accuracy)



def main():
    train_inputs, train_labels, test_inputs, test_labels = get_data()
    print(train_inputs.shape)
    num_classes = 4
    train_labels =tf.one_hot(train_labels, num_classes)
    test_labels = tf.one_hot(test_labels, num_classes)
    
    model = model_boi(num_classes)

    total_epochs = 100
    for e in range(total_epochs):
        print("E: ", e)
        train(model, train_inputs, train_labels)

    acc = test(model, test_inputs, test_labels)

main()