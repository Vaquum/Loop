import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class Experiment:

    '''
    What it does? 
    - Takes in a train and test data, as well as a single-file model
    - Run a random search over a set of parameters
    '''

    def __init__(self,
                 train,
                 test,
                 validate,
                 single_file_model):    
        
        self.train = train
        self.test = test
        self.validate = validate
        self.params = single_file_model.params
        self.prep = single_file_model.prep
        self.model = single_file_model.model
        self.history = []

    def params(self):
        return self.params()    

    def prep(self):
        return self.prep(self.train, self.test)
    
    def model(self):
        return self.model(self.train, self.test, params=self.params)

    def run(self,
            metric,
            metric_minimize=False,
            n_permutations=10):
        
        # Limit logging to errors
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Handle model specific data preration
        x_train, y_train = self.prep(self.train, mode='train')
        x_test, y_test = self.prep(self.test, mode='train')
        
        # Run the actual experiment
        for i in tqdm(range(n_permutations)):
            
            # Generate a random permutation of the parameters
            permutation = self._generate_permutation()
            
            # Train the model
            _trained_model, history = self.model(x_train,
                                                 y_train,
                                                 x_test,
                                                 y_test,
                                                 model_params=permutation)

            # Get the index of the maximum validation accuracy
            if metric_minimize is False:
                idx = np.argmax(history.history[metric])
            else:
                idx = np.argmin(history.history[metric])

            # Get the validation accuracy and loss at the maximum validation accuracy index
            val_accuracy = history.history[metric][idx]
            val_loss = history.history['val_loss'][idx]

            # Append the results to the history
            self.history.append([val_accuracy, val_loss, permutation])

        # Get params for the best model
        best_model_idx = np.argmax([i[0] for i in self.history])
        best_model_params = self.history[best_model_idx][2]

        new_train = pd.concat([self.train, self.test], axis=0)
        
        x_train, y_train = self.prep(new_train, mode='train')
        x_test, y_test = self.prep(self.validate, mode='train')

        model, history = self.model(x_train,
                                    y_train,
                                    x_test,
                                    y_test,
                                    model_params=best_model_params)
        
        return best_model_params, model, history
    
    def _generate_permutation(self):
        
        out_dict = {}

        for key in self.params().keys():
            out_dict[key] = np.random.choice(self.params()[key])

        return out_dict
