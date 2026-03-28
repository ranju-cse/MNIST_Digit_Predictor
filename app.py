from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
app=FastAPI()

model=tf.keras.models.load_model('mnist.h5')


class InputData(BaseModel):
    data:list
    


@app.get('/')
def home():
    return {"message":"Mnist App is Running"}

@app.post('/predict')
def predict(inputData:InputData):
    img=np.array(inputData.data)
#Normalize
    img=img/255.0
 #Reshape
    img=img.reshape(1,28,28)
    
 
    prediction=model.predict(img)
    digit=digit = int(np.argmax(prediction))
    return {"Digit:",digit}

