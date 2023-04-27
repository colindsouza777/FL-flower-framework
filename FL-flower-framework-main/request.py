# import http.client
# import json
from params import SetParams

# conn = http.client.HTTPConnection("localhost", 18400)

# headersList = {
#  "Accept": "*/*",
#  "User-Agent": "Thunder Client (https://www.thunderclient.com)",
#  "Authorization": "Basic Yml0Y29pbjpiaXRjb2lu",
#  "Content-Type": "application/json" 
# }

# payload = json.dumps({
#   "jsonrpc": "2.0",
#   "id": 1,
#   "method": "listtransactions",
#    "params": []
# }
# )

# conn.request("POST", "/wallet/akmore", payload, headersList)
# response = conn.getresponse()
# result = response.read()

# print(result.decode("utf-8"))
def getResult():
  result = SetParams('data')
  return result