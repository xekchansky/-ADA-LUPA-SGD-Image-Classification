import tensorflow as tf
import horovod.tensorflow.keras as hvd
from ftplib import FTP
import pandas as pd
import numpy as np
from matplotlib.image import imread
import os
from sklearn import preprocessing
import sys


def download_image(row, ftp, folder):
    ftp.cwd(str(row['dir']) + '/')
    filename = str(row['id']) + '.jpg'
    path = os.path.abspath(os.getcwd()) + '/' + folder + '/' + filename
    if not os.path.exists(path):
        ftp.retrbinary("RETR " + filename, open(path, "wb").write)
    ftp.cwd('../')


mnist = False

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

if mnist == True:
    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                 tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(100000000).batch(128)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


else:
    folder = 'Data'
    seed = hvd.rank()
    shard_size = 256 #8192

    try:
        os.mkdir(folder)
    except:
        pass

    # connect
    ftp = FTP('kur-serv.isa.ru')
    ftp.login()
    # go to data directory
    ftp.cwd('dataset')

    try:
        full_df = pd.read_csv(r'fragments.csv')
    except:
        filename = 'fragments.csv'
        with open(filename, "wb") as file_handle:
            ftp.retrbinary("RETR " + filename, file_handle.write)
        full_df = pd.read_csv(r'fragments.csv')

    # generate shard
    np.random.seed(seed)
    chosen_idx = np.random.choice(len(full_df), replace=True, size=shard_size)
    shard_df = full_df.iloc[chosen_idx]

    path = os.getcwd() + '/' + folder
    images = []

    ftp.cwd('dataset_fragments')

    done = 0
    for i, row in shard_df.iterrows():
        done += 1
        image_id = row['id']
        download_image(row, ftp, folder)
        image = imread(path + '/' + str(image_id) + '.jpg')
        images.append(image)
        sys.stdout.write('\r' + "images: " + str(done) + "/" + str(shard_size))
    shard_df['image'] = images
    df = shard_df.drop(['dir', 'original_image', 'temperature', 'angle'], axis=1)

    # preprocess_data
    X = df['image'].copy()
    X = np.stack(X).reshape((X.shape[0], 256, 256))
    y = df['additive'].copy()
    le = preprocessing.LabelEncoder()
    le.fit(['cGP', 'hG', 'cT', 'hM', 'hclear'])
    y = le.transform(y)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(y, tf.int64))
    )
    dataset = dataset.repeat().shuffle(10).batch(128)

    img_height = 256
    img_width = 256

    model = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
        tf.keras.layers.Conv2D(32, [5, 5], activation='relu', input_shape=(img_height, img_width, 1)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax'),
    ])


# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)

print(model.summary())

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
model.fit(dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)