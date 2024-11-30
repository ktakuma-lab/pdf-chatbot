#!/usr/bin/env python
# coding: utf-8

# Open a new Jupyter Notebook and import the required libraries to create an API using Flask, as well as the libraries to load the saved model:

# In[7]:


import flask
from flask import request
import torch
import final_model


# In[8]:


# Initialize the Flask app:
app = flask.Flask(__name__)
app.config["DEBUG"] = True


# In[10]:


# Define a function that loads the saved model and then instantiate the model:

def load_model_checkpoint(path):
    checkpoint = torch.load(path)
    model = final_model.Classifier(checkpoint["input_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model

model = load_model_checkpoint("checkpoint.pth")


# In[11]:


# Define the route of the API to /prediction and set the method to POST.
# Then, define the function that will receive the POST data and feed it to the model to perform a prediction:
@app.route('/prediction', methods=['POST'])
def prediction():
    body = request.get_json()
    example = torch.tensor(body['data']).float()
    pred = model(example)
    pred = torch.exp(pred)
    _, top_class_test = pred.topk(1, dim=1)
    top_class_test = top_class_test.numpy()
    return {"status":"ok", "result":int(top_class_test[0][0])}


# In[ ]:


# Run the Flask app:
#app.run(debug=True, use_reloader=False)

