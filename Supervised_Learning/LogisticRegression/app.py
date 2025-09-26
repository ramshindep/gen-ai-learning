from fastapi import FastAPI
import joblib
import numpy as np

model=joblib.load('flower_model.pkl')
encoder=joblib.load('flower_encoder.pkl')

app=FastAPI()

@app.get('/predict')
def prediction(sepal_length:float,sepal_width:float,petal_length:float,petal_width:float):
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    output=model.predict(input_data)
    predicted_species=encoder.inverse_transform(output)
    return{
        "predicted_species":predicted_species[0]
        }

#http://127.0.0.1:8000/predict?sepal_length=6.6&sepal_width=2.9&petal_length=4.6&petal_width=1.3