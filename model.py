import tensorflow as tf

class model_boi(tf.keras.Model):
    def __init__(self, num_classes):
        super(model_boi, self).__init__() 

        self.num_classes = num_classes
        self.batch_size = 128
        self.h1 = 500
        self.h2 = 300
        self.h3 = 200
        self.h4 = 100
        
        self.dense1 = tf.keras.layers.Dense(self.h1, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.h2, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(self.h3, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(self.h3, activation = 'relu')
        self.dense5 = tf.keras.layers.Dense(self.h4)
        self.dense6 = tf.keras.layers.Dense(self.num_classes)
        self.scce = tf.keras.losses.categorical_crossentropy
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    
    @tf.function
    def call(self, inputs):
        logits = self.dense1(inputs)
        logits = self.dense2(logits)
        logits = self.dense3(logits)
        # logits = self.dense4(logits)
        logits = self.dense5(logits)
        logits = self.dense6(logits)
        # print("logits shape", logits.shape)
        return logits

    def loss(self, logits, labels):
        loss_output = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.reduce_mean(loss_output)
        return loss


    def accuracy2(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


