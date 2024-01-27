from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from bson import ObjectId  # Import ObjectId from bson module
from pymongo import MongoClient

app = Flask(__name__)

# Load data from the provided data in tab-separated format
try:
    data = pd.read_csv('data.csv', delimiter='\t', encoding='utf-8', header=None, na_values=['', 'NA'])
except UnicodeDecodeError:
    try:
        # Try a different encoding
        data = pd.read_csv('data.csv', delimiter='\t', encoding='latin1', header=None, na_values=['', 'NA'])
    except UnicodeDecodeError:
        # Try another encoding
        data = pd.read_csv('data.csv', delimiter='\t', encoding='ISO-8859-1', header=None, na_values=['', 'NA'])

# Split the comma-separated columns into separate columns
data_split = data[0].str.split(',', expand=True)

# Assign appropriate column names
data_split.columns = ['Distance', 'Terrain', 'Weather', 'Speed']

# Convert columns to appropriate data types
data_split['Distance'] = pd.to_numeric(data_split['Distance'], errors='coerce')
data_split['Weather'] = pd.to_numeric(data_split['Weather'], errors='coerce')
data_split['Terrain'] = data_split['Terrain'].astype(str)
data_split['Speed'] = pd.to_numeric(data_split['Speed'], errors='coerce')

# Drop rows with missing values
data_split.dropna(inplace=True)

# Separate features (X) and target variable (y)
X = data_split[['Distance', 'Terrain', 'Weather']]
y = data_split['Speed']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate the maximum speed in the dataset for accuracy calculation
max_speed = data_split['Speed'].max()

# Initialize MongoDB client
client = MongoClient('mongodb://localhost:27017/')

# Specify the database and collection
db = client['roadrunner_data']
collection = db['prediction_results']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the HTML form
        distance = float(request.form['distance'])
        terrain = float(request.form['terrain'])
        weather = float(request.form['weather'])

        # Predict Roadrunner's speed using the trained linear regression model
        predicted_speed = model.predict([[distance, terrain, weather]])[0]

        # Calculate the accuracy as a percentage
        accuracy_percentage = (1 - abs(predicted_speed - max_speed) / max_speed) * 100

        # Store the inputs, prediction results, and input values in MongoDB
        prediction_result = {
            'predicted_speed': predicted_speed,
            'accuracy_percentage': accuracy_percentage,
            'inputs': {
                'distance': distance,
                'terrain': terrain,
                'weather': weather
            }
        }

        # Insert the document into the MongoDB collection
        result = collection.insert_one(prediction_result)

        # Render the prediction result in the HTML template
        return render_template('prediction.html', prediction=prediction_result)

    return render_template('index.html', predicted_speed=None)
@app.route('/all_predictions', methods=['GET'])
def all_predictions():
    # Retrieve all prediction results from MongoDB
    all_predictions = collection.find()
    return render_template('all_predictions.html', predictions=all_predictions)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
