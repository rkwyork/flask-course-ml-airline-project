import pandas as pd
import joblib

from flask import Flask, url_for, render_template, redirect
from forms import InputForm


app= Flask(__name__)

# configuration
app.config['SECRET_KEY'] =  'secret_key'

model = joblib.load('rf_model.joblib')

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title = 'Home')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        new_df= pd.DataFrame( dict(
            airline= form.airline.data,
            date_of_journey= form.date_of_journey.data.strftime('%Y-%m-%d'),
            source= form.source.data,
            destination= form.destination.data,
            dep_time= form.dep_time.data.strftime('%H:%M:%S'),
            arrival_time= form.arrival_time.data.strftime('%H:%M:%S'),
            duration= form.duration.data,
            total_stops= form.total_stops.data,
            additional_info= form.additional_info.data
        ), index=[0])
        predicted_price = model.predict(new_df)[0]
        message= f'Predicted price is {predicted_price:,.0f} INR.'
    else:
        message= 'Please provide valid input details.'

    return render_template('predict.html', title = 'Predict', form = form, output= message)



if __name__ == '__main__':
    app.run(debug=True)