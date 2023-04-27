import warnings
import flwr as fl
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from params import SetParams
import os.path

import utils

if __name__ == "__main__":
    # def trainClient():
        # Load MNIST dataset from https://www.openml.org/d/554
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
        def __init__(self,model):
            self.model = model
            print("I am here")
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(self.model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(self.model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            print(parameters)
            return utils.get_model_parameters(self.model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(self.model, parameters)
            loss = log_loss(y_test, self.model.predict_proba(X_test))
            accuracy = self.model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    first_iteration = True
    path = './model.pkl'
    check_file = os.path.isfile(path)
    # Start Flower client
    while(True):
        if not check_file:
            # first_iteration = False
            continue
        else:
            client = TrainModel(model)
            fl.client.start_numpy_client(server_address="localhost:8080", client=client)
            # pickle the last averaged model
            pickle.dump(model, open('model.pkl', 'wb'))
            break