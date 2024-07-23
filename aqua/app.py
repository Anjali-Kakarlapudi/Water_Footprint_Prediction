from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import mysql.connector
import sys
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)
db = mysql.connector.connect(host="localhost",user="root",password="",database="foot" )

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup')
def signup():
    return render_template('login.html',msg='')
   
@app.route('/userregister')
def userregister():
    return render_template('registration.html')

@app.route('/about')
def aboutus():
    return render_template('about.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/DailyHousehold')
def DailyHousehold():
    return render_template('dailyhousehold.html')

@app.route('/cropwater')
def cropwater():
    return render_template('cropwaterproof.html')

@app.route('/waterquality')
def waterquality():
    return render_template('waterqualityprints.html')

@app.route('/userregisterdb', methods=['POST'])
def do_userregisterdb():
    email=request.form['email']
    cpwd=request.form['cpwd']
    password=request.form['pwd']
    
    if password==cpwd:
        cursor = db.cursor()
        cursor.execute('insert into users(email,password) values("%s", "%s")' % \
             (email,password))
        db.commit()
        return render_template('login.html',msg='Registered')
    else:
        return render_template('login.html',msg='Password and confirm Password Not match')

@app.route('/predictDaily', methods=['GET'])
def predictDaily():
    
    # Load the dataset
    data = pd.read_csv('preprocessed_example.csv')

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the dataframe by date
    data = data.sort_values('Date')

    # Normalize the 'Consumption' column
    scaler = MinMaxScaler()
    data['Consumption'] = scaler.fit_transform(data[['Consumption']])

    # Function to create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # Define sequence length
    sequence_length = 10

    # Create sequences
    X, y = create_sequences(data['Consumption'].values, sequence_length)

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define LSTM model
    model = Sequential([
        LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Future prediction
    # Select a future date for prediction
    doc=request.args['doc']
    future_date = pd.to_datetime(doc)

    # Extract the last sequence of consumption data
    last_sequence = data['Consumption'].values[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)

    # Predict the consumption for the future date
    future_consumption = model.predict(last_sequence)
    print(f'Predicted Consumption for {future_date}: {scaler.inverse_transform(future_consumption)}')
    x=f'Predicted Consumption for {future_date}: {scaler.inverse_transform(future_consumption)}'
    return render_template("predictdailyhousehold.html",wp=x)


@app.route('/waterqualityprediction', methods=['GET'])
def waterqualityprediction():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    # Load the dataset
    data = pd.read_csv('waterQuality1.csv')

    # Drop rows with missing values
    data.dropna(inplace=True)
    # Convert object columns to numeric type
    object_columns = data.select_dtypes(include=['object']).columns
    data[object_columns] = data[object_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Encode categorical target variable
    label_encoder = LabelEncoder()
    data['is_safe'] = label_encoder.fit_transform(data['is_safe'])

    # Split data into features and target variable
    X = data.drop(columns=['is_safe'])
    y = data['is_safe']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost classifier
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Future prediction for new data
    # Example new data point
    t1=float(request.args['t1'])
    t2=float(request.args['t2'])
    t3=float(request.args['t3'])
    t4=float(request.args['t4'])
    t5=float(request.args['t5'])
    t6=float(request.args['t6'])
    t7=float(request.args['t7'])
    t8=float(request.args['t8'])
    t9=float(request.args['t9'])
    t10=float(request.args['t10'])
    t11=float(request.args['t11'])
    t12=float(request.args['t12'])
    t13=float(request.args['t13'])
    t14=float(request.args['t14'])
    t15=float(request.args['t15'])
    t16=float(request.args['t16'])
    t17=float(request.args['t17'])
    t18=float(request.args['t18'])
    t19=float(request.args['t19'])
    t20=float(request.args['t20'])
    new_data_point = pd.DataFrame({
        'aluminium': [t1],
        'ammonia': [t2],
        'arsenic': [t3],
        'barium': [t4],
        'cadmium': [t5],
        'chloramine': [t6],
        'chromium': [t7],
        'copper': [t8],
        'flouride': [t9],
        'bacteria': [t10],
        'viruses': [t11],
        'lead': [t12],
        'nitrates': [t13],
        'nitrites': [t14],
        'mercury': [t15],
        'perchlorate': [t16],
        'radium': [t17],
        'selenium': [t18],
        'silver': [t19],
        'uranium': [t20]
    })

    # Make prediction for the new data point
    prediction = model.predict(new_data_point)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    fr=""
    if prediction_label==0:
        fr="Not Safe"
    else:
        fr="Safe"
    print(f'Prediction for new data point: {prediction_label}')
    x=f'Prediction for new data point: {fr}'
    return render_template("predictdailyhousehold.html",wp=x)


@app.route('/predictWaterRequirement', methods=['GET'])
def predictWaterRequirement():
    data = pd.read_csv('datasetfinal.csv')

    # Preprocess data
    stype = {'DRY': 0, 'HUMID': 1, 'WET': 2}
    data['SOILTYPE'] = data['SOILTYPE'].map(stype)

    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    scaler = MinMaxScaler()
    data[['SOILTYPE', 'temperature', 'humidity', 'ph']] = scaler.fit_transform(data[['SOILTYPE', 'temperature', 'humidity', 'ph']])

    # Split data into features and target variable
    X = data[['SOILTYPE', 'temperature', 'humidity', 'ph', 'label']]
    y = data['WATERREQUIREMENT']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape input data for LSTM
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define LSTM model
    model = Sequential([
        LSTM(units=64, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        Dense(units=128, activation='relu'),
        Dense(units=1)  # For regression, we use a single neuron without activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(X_test_reshaped, y_test)
    print(f'Test Loss: {test_loss}')

    # Predict on test data
    y_pred = model.predict(X_test_reshaped)

    # Calculate and print mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')

    # Calculate and print mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Concatenate the predictions with actual values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    print(results)

    #added here
    '''
    soil_type = 'DRY'
    temperature = 21.77046169
    humidity = 80.31964408
    ph = 7.038096361
    crop_name = 'rice'  # Include crop name
    '''
    # Take input from the user
    SOILTYPE=request.args['soiltype']
    temperature=request.args['temp']
    humidity=request.args['humidity']
    
    ph=request.args['ph']
    cropname=request.args['cropname']

    # Assuming you have a new data point stored in a DataFrame called 'new_data'
    new_data = pd.DataFrame([[SOILTYPE, temperature, humidity, ph,cropname]], columns=['SOILTYPE', 'temperature', 'humidity', 'ph','label'])
    # Preprocess the new data point
    new_data['SOILTYPE'] = new_data['SOILTYPE'].map(stype)
    new_data['label'] = label_encoder.transform(new_data['label'])
    new_data[['SOILTYPE', 'temperature', 'humidity', 'ph']] = scaler.transform(new_data[['SOILTYPE', 'temperature', 'humidity', 'ph']])

    # Reshape input data for LSTM
    X_new_reshaped = new_data.values.reshape((1, 1, new_data.shape[1]))

    # Predict water requirement for the new data point
    water_requirement_pred = model.predict(X_new_reshaped)

    # Print the predicted water requirement
    print("Predicted Water Requirement for the new data point:")
    return render_template("predictresult.html",wp=water_requirement_pred[0][0])




    
@app.route('/login', methods=['POST'])
def do_login():
    flag=False
    cursor = db.cursor()
    username=request.form['email']
    password=request.form['password']
    sql = "SELECT * FROM users WHERE email= '%s' and password = '%s' " % (username,password)
    print("Sql  is ",sql)
    cursor.execute(sql)
    rows_count = cursor.fetchall()
    if len(rows_count) > 0:
        session['logged_in'] = True
        session['uid'] = username
        flag=True
    else:
        flag=False
    if flag:
        return render_template('prediction.html')
    else:
        return render_template('login.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()

 
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
