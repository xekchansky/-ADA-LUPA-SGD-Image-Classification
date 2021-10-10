from ftplib import FTP
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, initializers

def download_image(row, ftp, folder):
    ftp.cwd(str(row['dir']) + '/')
    filename = str(row['id']) + '.jpg'
    path = os.path.abspath(os.getcwd()) + '\\' + folder + '\\' + filename
    if not os.path.exists(path):
        ftp.retrbinary("RETR " + filename, open(path, "wb").write)
    ftp.cwd('../')

def get_data(df, total_files=4559688, batch_size=10, seed=228):
    os.mkdir('Data')
    np.random.seed(seed)
    chosen_idx = np.random.choice(len(df), replace = True, size = batch_size)
    df2 = df.iloc[chosen_idx]
    df2.apply(lambda row: download_image(row), axis = 1)
    path = os.getcwd() + '\\Data'
    for image_id in df2['id']:
        image_name = str(image_id) + '.jpg'
        image = imread(path + '\\' + image_name)

def make_model(learning_rate=None, regularization_rate=None, seed=0):
    img_height = 256
    img_width = 256

    initializer = initializers.HeNormal(seed=seed)
    
    if learning_rate is None:
        optimizer = tf.keras.optimizers.Adam()
    else:
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    
    if regularization_rate is None:
        kernel_regularizer = None
        dense_kernel_regilarizer = None
    else:
        kernel_regularizer = tf.keras.regularizers.L2(regularization_rate)
        dense_kernel_regilarizer = tf.keras.regularizers.L2(regularization_rate)
    activity_regularizer = None #tf.keras.regularizers.L2(0.01)
    dense_activity_regularizer = None
    
    model = models.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)))

    model.add(layers.Conv2D(16, (5, 5), activation='relu', 
                            input_shape=(img_height, img_width, 1), 
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu',
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            kernel_initializer=initializer, 
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=dense_kernel_regilarizer,
                           activity_regularizer=dense_activity_regularizer))
    model.add(layers.Dense(5))

    model.compile(optimizer=optimizer,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          #loss = tf.losses.softmax_cross_entropy(),
          #loss = tf.compat.v1.losses.softmax_cross_entropy(),
          #loss = 'categorical_crossentropy',
          metrics=['accuracy'])
    return model
