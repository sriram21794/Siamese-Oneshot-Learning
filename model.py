import tensorflow as tf

from dataset import get_tf_dataset
import datetime
import os
from config import config

def mkdir_(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


class SiameseModel():
    def __init__(self):
        self.model = None
        self.build()

    def add_conv2d_block(self, net, filters, kernel_size, add_maxpool=False):
        net.add(tf.keras.layers.Conv2D( 
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, mean=0.0, seed=config.seed),
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, mean=0.5, seed=config.seed),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))

        if add_maxpool:
            net.add(tf.keras.layers.MaxPooling2D())

    def get_callbacks(self):

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=config.log_dir)
        ]

        return callbacks


    def build(self):
        print("Building Siamese Network.")
        input_1 = tf.keras.layers.Input(config.target_size, name="input_1")
        input_2 = tf.keras.layers.Input(config.target_size, name="input_2")

        net = tf.keras.models.Sequential()
        
        self.add_conv2d_block(net, filters=64, kernel_size=(10, 10), add_maxpool=True)
        self.add_conv2d_block(net, filters=128, kernel_size=(7, 7), add_maxpool=True)
        self.add_conv2d_block(net, filters=128, kernel_size=(4, 4), add_maxpool=True)
        self.add_conv2d_block(net, filters=256, kernel_size=(4, 4), add_maxpool=False)
        
        net.add(tf.keras.layers.Flatten())

        net.add(tf.keras.layers.Dense(
            units=4096,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.2, mean=0.0, seed=config.seed),
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, mean=0.5, seed=config.seed),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))

        twin_1 = net(input_1)
        twin_2 = net(input_2)

        
        distance_layer = tf.keras.layers.Lambda(
            lambda inps: tf.abs(inps[0] - inps[1]))

        l1_distance = distance_layer([twin_1, twin_2])

        output = tf.keras.layers.Dense(1, activation='sigmoid')(l1_distance)

        self.model = tf.keras.models.Model(inputs=[input_1, input_2 ], outputs=output)
        self.model.summary()

    def train(self, background_directory, evaluation_directory, model_path):
    
        background_tf_dataset = get_tf_dataset(background_directory, use_augmentation=True)
        evaluation_tf_dataset = get_tf_dataset(evaluation_directory, use_augmentation=True)

        start_ = datetime.datetime.now()
        print(f"Train Started at {start_}")

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer="sgd")
    
        self.model.fit(
            background_tf_dataset,
            steps_per_epoch=config.background_samples // config.batch_size // 2, 
            validation_data=evaluation_tf_dataset,
            validation_steps=config.evaluation_samples // config.batch_size // 2,
            callbacks=self.get_callbacks(),
            epochs=config.nb_epochs
        )
        end_ = datetime.datetime.now()   
        print(f"Train Ended at {end_}. Total Time = {end_ - start_}")
        print(f"Saving Model to {model_path}")
        self.model.save(model_path)
        










        
        

        






    
