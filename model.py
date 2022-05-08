import tensorflow as tf

class model_boi(tf.keras.Model):
    def __init__(self, num_classes):
        super(model_boi, self).__init__() 

        self.num_classes = num_classes
        self.batch_size = 20
        self.h1 = 19
        self.h2 = 19
        self.h3 = 19
        self.h4 = 19
        
        self.dense1 = tf.keras.layers.Dense(self.h1, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.h2, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(self.h3, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(self.h4, activation = 'relu')
        self.dense5 = tf.keras.layers.Dense(self.num_classes, activation = 'softmax')
        self.scce = tf.keras.losses.categorical_crossentropy
        self.lr = 1e-2
        self.optimizer = tf.keras.optimizers.Adam(self.lr)


    @tf.function
    def call(self, inputs):
        output = self.dense1(inputs)
        logits = self.dense2(output)
        logits = self.dense3(logits)
        logits = self.dense4(logits)
        probs = self.dense5(logits)
        return probs

    def loss(self, prbs, labels):
        # print(labels.shape)
        loss = self.scce(labels, prbs)
        loss_sum = tf.reduce_sum(loss)
        return loss_sum

    def accuracy(self, prbs, labels):
        # print(prbs)
        decoded_symbols = tf.argmax(input=prbs, axis=1)
        labels = tf.argmax(labels, axis=1)
        # print(decoded_symbols, labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32))
        return accuracy


