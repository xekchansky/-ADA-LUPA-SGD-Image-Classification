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

folder = 'Data'
seed = 0
shard_size = 8192

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
    sys.stdout.write('\r' + str(done))
    done += 1
    image_id = row['id']
    download_image(row, ftp, folder)
    image = imread(path + '/' + str(image_id) + '.jpg')
    images.append(image)
shard_df['image'] = images
df = shard_df.drop(['dir', 'original_image',  'temperature', 'angle'], axis=1)

# preprocess_data
X = df['image'].copy()
X = np.stack(X).reshape((X.shape[0], 256, 256, 1))
y = df['additive'].copy()
le = preprocessing.LabelEncoder()
le.fit(['cGP', 'hG', 'cT', 'hM', 'hclear'])
y = le.transform(y)

X_train = X
y_train = y
