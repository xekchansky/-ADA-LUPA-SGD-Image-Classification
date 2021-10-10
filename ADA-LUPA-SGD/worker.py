import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import pandas as pd
import shutil
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import backend

import utils

class Worker:
    def __init__(self, shard_size, seed):
        self.seed = seed
        self.shard_size = shard_size
        self.model = None
        self.df = None #shard data frame
        self.le = None #Label Encoder
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def get_data(self, download=False, folder='Data'):
        if download:
            #connect
            ftp = FTP('83.149.249.48')
            ftp.login()
            
            #go to data directory
            ftp.cwd('dataset')
            
            #clean old data
            shutil.rmtree(folder, ignore_errors=True)
            
            #create data folder
            os.mkdir(folder)
            
            #try open fragments.csv, download if does not exist
            try:
                full_df = pd.read_csv (r'fragments.csv')
            except:
                filename = 'fragments.csv'
                with open(filename, "wb") as file_handle:
                    ftp.retrbinary("RETR " + filename, file_handle.write)
                full_df = pd.read_csv (r'fragments.csv')
                
            #generate shard
            np.random.seed(self.seed)
            chosen_idx = np.random.choice(len(full_df), replace = True, size = self.shard_size)
            shard_df = full_df.iloc[chosen_idx]
            
            #download images
            shard_df.apply(lambda row: download_image(row, ftp, folder), axis = 1)
        else:
            #open fragments.csv
            full_df = pd.read_csv (r'fragments.csv')
            
            #generate shard
            np.random.seed(self.seed)
            chosen_idx = np.random.choice(len(full_df), replace = True, size = self.shard_size)
            shard_df = full_df.iloc[chosen_idx]
        
        #load images
        path = os.getcwd() + '\\' + folder
        images = []
        for image_id in shard_df['id']:
            image = imread(path + '\\' + str(image_id) + '.jpg')
            images.append(image)
        shard_df['image'] = images
        self.df = shard_df.drop(['dir', 'original_image',  'temperature', 'angle'], axis=1)
        
    def preprocess_data(self):
        X = self.df['image'].copy()
        X = np.stack(X).reshape((X.shape[0], 256, 256, 1))
        y = self.df['additive'].copy()
        self.le = preprocessing.LabelEncoder()
        self.le.fit(['cGP', 'hG', 'cT', 'hM', 'hclear'])
        y = self.le.transform(y)

        self.X_train = X
        self.y_train = y
        
    def fit_model(self, num_round, Mu, a, local_steps=1, minibatch_size=128):
        X_train, y_train = shuffle(self.X_train, self.y_train, random_state=self.seed+num_round)
        for i in range(local_steps):
            #get minibutch
            X_batch = X_train[i*minibatch_size:i*minibatch_size + minibatch_size]
            y_batch = y_train[i*minibatch_size:i*minibatch_size + minibatch_size]
            
            #update learning rate
            learning_rate = 4 / (Mu * (num_round * local_steps + a))
            backend.set_value(self.model.optimizer.learning_rate, learning_rate)
            
            self.model.train_on_batch(X_batch, y_batch)
        
    def save_weights(self):
        self.model.save_weights('./checkpoints/my_checkpoint' + str(self.shard_size) + '_' + str(self.seed))
        
    def load_weights(self):
        self.model.load_weights('./checkpoints/my_checkpoint' + str(self.shard_size) + '_' + str(self.seed))
        
    def test_model(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print("Model, accuracy: {:5.2f}%".format(100 * acc))
        
    def plot_history(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
            
    def run(self):
            return 0
