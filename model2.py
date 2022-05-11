import tensorflow as tf

class lin_reg(tf.keras.Model):
    def __init__(self):
        super(lin_reg, self).__init__() 

        self.batch_size = 60
        self.h1 = 500
        self.h2 = 200
        self.h3 = 100
        self.h4 = 50

        self.h5 = 16
        
        
        self.dense1 = tf.keras.layers.Dense(self.h5, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.h5, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(self.h5, activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(self.h5, activation = 'relu')
        # self.dense5 = tf.keras.layers.Dense(self.h5, activation = 'relu')
        self.dense6 = tf.keras.layers.Dense(1)
        self.loss_func = tf.keras.losses.MeanSquaredError()
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
    

    @tf.function       
    def call(self, inputs):
        output = self.dense1(inputs)
        logits = self.dense2(output)
        logits = self.dense3(logits)
        logits = self.dense4(logits)
        # logits = self.dense5(logits)
        logits = self.dense6(logits)
        print("logits shape", logits.shape)
        return logits

    def loss(self, pred, labels):
        loss = self.loss_func(labels, pred)
        return loss

    
