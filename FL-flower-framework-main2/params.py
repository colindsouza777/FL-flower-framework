import pandas as pd
from utils import transform

def SetParams(data):
    response = transform(data)
    # print(response)
    return response


# SetParams("data")