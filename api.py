import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')

    final_features = [features]

    final_features_df = pd.DataFrame(final_features, columns=['City', 'type', 'Area', 'Street', 'number_in_street', 'city_area', 'floor', 'num_of_images',
                                                               'hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'condition ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ',
                                                               'handicapFriendly ', 'entranceDate ', 'furniture ', 'publishedDays '])
    # Convert all other values to strings
    final_features_df = final_features_df.astype(str)
    
    # Convert 'num_of_images' to float
    try:
        final_features_df['num_of_images'] = final_features_df['num_of_images'].astype(float)
    except ValueError:
        return render_template('index.html', prediction_text='Invalid data type for "num_of_images"')

    prediction = rf_model.predict(final_features_df)

    return render_template('index.html', prediction_text='Asset predicted price is: {} ₪' .format(prediction))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
