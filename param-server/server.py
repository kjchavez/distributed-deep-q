from flask import Flask, jsonify
from barista import messaging
from barista import baristanet
app = Flask(__name__)

@app.route("/")
def hello():
    return "Param Server"

@app.route('/api/v1/latest_model', methods=['GET'])
def get_model_params():
  # assuming the return message would be a string(of bytes)
  m = messaging.create_message(centralModel, compress=False)
  return Response(m, status=200)

@app.route('/api/v1/update_model/', methods=['POST'])
def update_params():

  updateParams = messaging.load_gradient_message(request.data, compressed = False)
  update(params)
  return "model updated"

def update(params):
  return 

def initParams():
  global centralModel
  centralModel = baristanet.BaristaNet()
  return  

if __name__ == "__main__":
    initParams()
    app.run(debug=True, port=5500)