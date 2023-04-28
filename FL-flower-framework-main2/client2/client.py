import warnings
import flwr as fl
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from params import SetParams
import os.path
from request import getResult

import utils

if __name__ == "__main__":
    # def trainClient():
        # Load MNIST dataset from https://www.openml.org/d/554
    path = './model.pkl'
    check_file = os.path.isfile(path)
    
    (X_train, y_train), (X_test, y_test) = utils.load_transaction_data(None)

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)
    
    
        


    class TrainModel(fl.client.NumPyClient):
        def __init__(self,model, X, Y):
            self.model = model
            self.X = X
            self.Y = Y
            print("I am here")
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(self.model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(self.model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.X, self.Y)
            print(f"Training finished for round {config['server_round']}")
            print(parameters)
            return utils.get_model_parameters(self.model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            # utils.set_model_params(self.model, parameters)
            # loss = log_loss(y_test, self.model.predict_proba(X_test))
            # # accuracy = self.model.score(X_test, y_test)
            # return loss, len(X_test), {"accuracy": accuracy}
            pass

    first_iteration = True
    # Start Flower client
    
    while(True):
        if not check_file:
            X, Y, _, _ = utils.load_transaction_data(None)
            client = TrainModel(model, X, Y)
            fl.client.start_numpy_client(server_address="localhost:8080", client=client)
            pickle.dump(model, open('model.pkl', 'wb'))  
        else:
            # (X_train, y_train), (X_test, y_test) = getResult()
            X = getResult()
            # # Split train set into 10 partitions and randomly use one for training.
            # partition_id = np.random.choice(10)
            # (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

            # Create LogisticRegression Model
            model = LogisticRegression(
                penalty="l2",
                max_iter=1,  # local epoch
                warm_start=True,  # prevent refreshing weights when fitting
            )

            # Setting initial parameters, akin to model.compile for keras models
            utils.set_initial_params(model)
            model = pickle.load(path)
            result = model.predict(X)
            # (X_train, y_train), (X_test, y_test) = X, Y
            X , Y = X, result
            client = TrainModel(model, X, Y)
            fl.client.start_numpy_client(server_address="localhost:8080", client=client)
            # pickle the last averaged model
            pickle.dump(model, open('model.pkl', 'wb'))
            break