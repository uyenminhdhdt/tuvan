#//////////////////
from flask import Flask, render_template,request
import numpy as np

import pickle#Initialize the flask App

app = Flask(__name__)
#Mở mô ình dự báo
model = pickle.load(open("classifier.pkl", "rb"))

#default page of our web-app
app=Flask(__name__,template_folder='d:/DATA')




#/////
@app.route('/')
def home():
    
    return render_template('gd.html')

#/To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2) 
    return render_template('gd.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))
    
if __name__ == "__main__":
    app.run(debug=True)

import gunicorn
