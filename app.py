from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

#FastAPI() → creates a web application instance
#app → this is the main object that handles our API
app=FastAPI()
#loads the saved model
model=tf.keras.models.load_model('mnist.h5')

#Defines what input your API expects 
class InputData(BaseModel):
    data:list
    


@app.get('/')
def home():
    return {"message":"Mnist App is Running"}

@app.post('/predict')
def predict(inputData:InputData):
    
    #Extract data
    img=np.array(inputData.data)
    #Normalize
    img=img/255.0
    #Reshape
    img=img.reshape(1,28,28)
    
 
    prediction=model.predict(img)
    digit=digit = int(np.argmax(prediction))
    return {"Digit":digit}

