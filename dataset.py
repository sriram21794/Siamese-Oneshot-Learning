import tensorflow as tf


import glob
import os
import numpy as np
import random

random.seed(22)



TARGET_SIZE = (150, 150, 1)
BATCH_SIZE = 64
WHITE_LIST_FORMATS = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')
NB_EPOCHS = 10


def process_images(images):
    
    inputs = []
    
    for image in images:
        img = tf.keras.preprocessing.image.load_img(image, target_size=TARGET_SIZE[:2], 
                                                    color_mode="rgb" if TARGET_SIZE[2] == 3 else 'grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # TODO: Add aug
        img_array = img_array/ 255.
        inputs.append(img_array)

    return np.array(inputs)
        
    
def pairwise_generator(categoriores, category_to_images, batch_size):
    
    for category, category_images in category_to_images.items():
        random.shuffle(category_images)
        
    random.shuffle(categoriores)
    
    images = []
    image_2_category = {}
    for category in categoriores:
        for image in category_to_images[category]:
            images.append(image)
            image_2_category[image] = category
            
    
    batch_image_pairs = []
    for image_1, image_2 in zip(images[::2], images[1::2]):
        batch_image_pairs.append([image_1, image_2])
        batch_image_pairs.append([image_1, random.choice(images)])
        
        if len(batch_image_pairs) == batch_size:
            input_1 = process_images([ image_pair_1 for image_pair_1, image_pair_2 in batch_image_pairs])
            input_2 = process_images([ image_pair_2 for image_pair_1, image_pair_2 in batch_image_pairs])

            target = np.array([1 if image_2_category[image_pair_1] == image_2_category[image_pair_2] else 0
                                for image_pair_1, image_pair_2 in batch_image_pairs])

            yield { "input_1": input_1, "input_2": input_2}, target


def get_tf_dataset(directory_, use_augmentation=False, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS):

    datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()

    alphabets = sorted(os.listdir(directory_))
    alphabets = alphabets[:2]
    
    alphabet_to_images = {}
    for alphabet_id, alphabet  in enumerate(alphabets):
        alphabet_to_images[alphabet] = list(filter( lambda _: _.rsplit(".", 1)[-1] in WHITE_LIST_FORMATS ,
                                                    glob.glob(f"{os.path.join(directory_, alphabet)}/**/*", recursive=True)))


    images = [image for images_per_alphabet in alphabet_to_images.values() for image in images_per_alphabet]

    print(f"Got {len(alphabets)} alphabets and  {len(images)} images")

    output_types = {"input_1": tf.float32, "input_2": tf.float32}, tf.float32
    
    output_shapes=({"input_1": tf.TensorShape((None, *TARGET_SIZE)), 
                    "input_2": tf.TensorShape((None, *TARGET_SIZE))}, 
                    tf.TensorShape((None,)))
    
    def dummy_generator(): # To avoid converting strings and lists to tensor
        for data in pairwise_generator(alphabets, alphabet_to_images, batch_size):
            yield data

    tf_dataset = tf.data.Dataset.from_generator(
        dummy_generator,
        output_types=output_types,
        output_shapes=output_shapes
        )

    tf_dataset = tf_dataset.repeat(nb_epochs) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset
    