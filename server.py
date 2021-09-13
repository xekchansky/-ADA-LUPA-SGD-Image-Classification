import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing

import utils
import worker

class Server:
    def __init__(self, num_workers=3, shard_size=8000, learning_rate=0.001, regularization_rate=0.0001, debug = False):
        self.num_workers = num_workers
        self.shard_size=shard_size
        self.nodes = []
        self.model = utils.make_model(learning_rate=learning_rate, regularization_rate=regularization_rate)
        if debug: self.model.summary()
        self.losses = []
        self.accuracies = []
        self.debug = debug
        for i in range(num_workers):
            if debug: print('Preparing worker', i)
            self.nodes.append(worker.Worker(shard_size=shard_size,  
                                            seed = i))
            
            if debug: print('Getting data')
            self.nodes[i].get_data(folder = 'Data' + str(shard_size) + '_' + str(i))
            
            if debug: print('Preprocessing')
            self.nodes[i].preprocess_data()
            
            if debug: print('Setting start model')
            self.nodes[i].model = self.model
            
            if debug: print('Setting initialization weights')
            for j in range(len(self.model.layers)):
                self.nodes[i].model.layers[j].set_weights(np.asarray(self.model.layers[j].get_weights()))
                
            if debug: print('Done\n')
                
        self.get_test_data()
        self.preprocess_test_data()
                
    def get_test_data(self, download=False, folder='Data'):
        #пока что тут костыль
        #тестируется на 300 изображениях из папок Data100_0, Data100_1, Data100_2
        
        full_df = pd.read_csv (r'fragments.csv')
        test_df = None
        for i in range(3):
            #generate shard
            np.random.seed(i)
            chosen_idx = np.random.choice(len(full_df), replace = True, size = 100)
            shard_df = full_df.iloc[chosen_idx]

            #load images
            path = os.getcwd() + '\\' + folder + '100_' + str(i)
            images = []
            for image_id in shard_df['id']:
                image = imread(path + '\\' + str(image_id) + '.jpg')
                images.append(image)
            shard_df['image'] = images
            if i == 0:
                test_df = shard_df.drop(['dir', 'original_image',  'temperature', 'angle'], axis=1)
            else:
                test_df = test_df.append(shard_df.drop(['dir', 'original_image',  'temperature', 'angle'], axis=1))
        self.test_df = test_df
        
    def preprocess_test_data(self):
        X = self.test_df['image'].copy()
        X = np.stack(X).reshape((X.shape[0], 256, 256, 1))
        y = self.test_df['additive'].copy()
        self.le = preprocessing.LabelEncoder()
        self.le.fit(['cGP', 'hG', 'cT', 'hM', 'hclear'])
        y = self.le.transform(y)
        
        self.X_test = X
        self.y_test = y
        
    def test_model(self):
        verbose = 0
        if self.debug: verbose = 2
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=verbose)
        self.losses.append(loss)
        self.accuracies.append(acc)
        if self.debug: print("Model, accuracy: {:5.2f}%".format(100 * acc))

    def run(self, Mu, a, rounds=10, epochs_per_round=1, steps_per_epoch=100, local_steps=9, minibatch_size=128):
        for r in range(rounds):
            #train workers and collect weights
            if self.debug: print('Start of round', r)
            if self.debug: print('learning rate:', 4 / (Mu * (r * local_steps + a)))
            for i in range(len(self.nodes)):
                if self.debug: print('Worker', i)
                self.nodes[i].fit_model(epochs=epochs_per_round, 
                                        steps_per_epoch=steps_per_epoch, 
                                        local_steps=local_steps, 
                                        minibatch_size=minibatch_size,
                                        num_round=r,
                                        Mu=Mu,
                                        a=a)
                self.nodes[i].save_weights()
                if i == 0:
                    for j in range(len(self.model.layers)):
                        w = np.asarray(self.nodes[i].model.layers[j].get_weights())/self.num_workers
                        self.model.layers[j].set_weights(w)
                else:
                    for j in range(len(self.model.layers)):
                        self.model.layers[j].set_weights(np.asarray(self.nodes[i].model.layers[j].get_weights()) + np.asarray(self.nodes[i].model.layers[j].get_weights())/self.num_workers)
            
            #update workers weights
            for i in range(len(self.nodes)):
                for j in range(len(self.model.layers)):
                    self.nodes[i].model.layers[j].set_weights(np.asarray(self.model.layers[j].get_weights()))
                    
            if self.debug: print('RESULTS OF ROUND', r)
            self.test_model()
            if self.debug: print()

        #save result
        self.model.save_weights('./checkpoints/my_checkpoint' + '_Global')
        
    def plot_history(self):
        plt.plot(self.losses, label='loss')
        plt.xlabel('Round of communication')
        plt.ylabel('Loss')
        #plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()

        plt.plot(self.accuracies, label='accuracy')
        plt.xlabel('Round of communication')
        plt.ylabel('Accuracy')
        plt.ylim([0.0, 1])
        plt.legend(loc='lower right')
        plt.show()