import tensorflow as tf

from dataset import get_tf_dataset
import datetime

TARGET_SIZE = (150, 150, 1)
BATCH_SIZE = 64
WHITE_LIST_FORMATS = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')
NB_EPOCHS = 10
SEED = 22


class SiameseModel():
    def __init__(self):
        # self.background_directory = background_directory
        # self.evaluation_directory = evaluation_directory
        # self.model_path = model_path
        self.model = None
        self.build()

    def add_conv2d_block(self, net, filters, kernel_size, add_maxpool=False):
        net.add(tf.keras.layers.Conv2D( 
            filters=filters,
            kernel_size=(10, 10),
            activation="relu",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, mean=0.0, seed=SEED),
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, mean=0.5, seed=SEED),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))

        if add_maxpool:
            net.add(tf.keras.MaxPooling2D())


    def build(self):
        print("Building Siamese Network.")
        input_1 = tf.keras.layers.Input(TARGET_SIZE, name="input_1")
        input_2 = tf.keras.layers.Input(TARGET_SIZE, name="input_2")

        net = tf.keras.models.Sequential()
        
        self.add_conv2d_block(net, filters=64, kernel_size=(10, 10), add_maxpool=True)
        self.add_conv2d_block(net, filters=128, kernel_size=(7, 7), add_maxpool=True)
        self.add_conv2d_block(net, filters=128, kernel_size=(4, 4), add_maxpool=True)
        self.add_conv2d_block(net, filters=256, kernel_size=(4, 4), add_maxpool=False)
        
        net.add(tf.keras.layers.Flatten())

        net.add(tf.keras.layers.Dense(
            units=4096,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.2, mean=0.0, seed=SEED),
            bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, mean=0.5, seed=SEED),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ))

        twin_1 = net(input_1)
        twin_2 = net(input_2)

        
        distance_layer = tf.keras.layers.Lambda(
            lambda t1, t2: tf.abs(t1 - t2))

        l1_distance = distance_layer([twin_1, twin_2])

        output = tf.keras.layers.Dense(1, activation='sigmoid')(l1_distance)

        self.model = tf.keras.models.Model(inputs=[input_1, input_2 ], outputs=output)

    def train(self, background_directory, evaluation_directory, model_path, epochs=NB_EPOCHS):
    
        background_tf_dataset = get_tf_dataset(background_directory, use_augmentation=True)
        evaluation_tf_dataset = get_tf_dataset(evaluation_directory, use_augmentation=True)

        start_ = datetime.datetime.now()
        print(f"Train Started at {start_}")

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer="sgd")

        self.model.fit(
            background_tf_dataset,
            validation_data=evaluation_tf_dataset,
        )
        









        
        

        






    
