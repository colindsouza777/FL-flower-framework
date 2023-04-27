import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple
from flwr.server.client_manager import ClientManager
from flwr.server.client_manager import SimpleClientManager

client_ids = []

def eval_result_callback(results):
    # print the results for each client
    for client_id, result in results.items():
        print(f"Evaluation result for client {client_id}: {result}")

# : Optional[EvaluationResult]
# -> Optional[Parameters]

def on_fit(client_id: str, parameters: fl.common.NDArrays, round_num: int) :
    # Do something with trained parameters
    return parameters

def on_evaluate(client_id: str, parameters: fl.common.NDArrays, eval_result, round_num: int) -> None:
    # Do something with evaluation result, including trained parameters
    if eval_result is not None:
        print(f"Client {client_id} evaluated with accuracy {eval_result.accuracy}")
    else:
        print("got model from client")

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_transaction_data(None)

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    
    first_iteration = True

    while True:

        if first_iteration:
            first_iteration = False
        else:
            model = LogisticRegression()
            utils.set_initial_params(model)
            
            strategy = fl.server.strategy.FedAvg(
                min_available_clients=2,
                evaluate_fn=get_evaluate_fn(model),
                on_fit_config_fn=fit_round
            )

            class CustomServer(fl.server.Server):

                def __init__(self,clientManger,stratergy):
                    super().__init__(client_manager=clientManger,strategy=strategy)

                def on_evaluate(
                    self,
                    rnd: int,
                    cid: str,
                    result: Tuple[float, int],
                    num_examples: int,
                    parameters: fl.common.NDArrays,
                ) :
                    print(f"Received evaluated model from client {cid}")
                    print("Here")
                    # Do something with the evaluated model here
                    # Return None to discard the evaluated model, or return a new weight
                    # to use it to update the global model.
                    return None
            simpleClientManager = SimpleClientManager()
            server = CustomServer(simpleClientManager,strategy)
            fl.server.start_server(server_address="[::]:8080", server = server,config=fl.server.ServerConfig(num_rounds=10))
