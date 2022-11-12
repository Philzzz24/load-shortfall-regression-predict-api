"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
# Data preprocessing.
prep_data = _preprocess_data(data)
def drop_columns(input_df):
    output_df = input_df.copy()
    for column in output_df:
        with_nulls = output_df[column].isna().sum()
        if with_nulls > 0:
            output_df = output_df.drop(column, axis = 1)
        else:
            if column not in df.select_dtypes(include='number').columns:
                output_df = output_df.drop(column, axis = 1)
    return output_df



# The new training dataset
clean_df = drop_columns(df)

# The new test dataset
clean_df_test = drop_columns(df_test)

"""    
From our findings in the Exploratory Data Analysis phase, the columns that require engineering are:

Unnamed: 0 : It is redundant
time : Convert it to datetime type
Valencia_wind_deg : Convert it to numeric type
Seville_pressure : Convert it to numeric type
Valencia_pressure : It has null values
Unnamed:  
"""
# Confirm redundancy

df['Unnamed: 0'].head()

# Function to drop "Unnamed: 0" column

def drop_unnamed(input_df):
    output_df = input_df.copy()
    output_df = output_df.drop(['Unnamed: 0'], axis = 1)
    return output_df


# Drop "Unnamed: 0" in training dataset

df_train_no_unnamed = drop_unnamed(df)
df_train_no_unnamed.head(2)


# Drop "Unnamed: 0" in training dataset

df_train_no_unnamed = drop_unnamed(df)
df_train_no_unnamed.head(2)


# Function to process date

def process_date(input_df):
    output_df = input_df.copy()
    output_df['time'] = pd.to_datetime(output_df['time'])
    output_df['hour'] = output_df['time'].dt.hour
    output_df['month'] = output_df['time'].dt.month
    output_df['year'] = output_df['time'].dt.year
    output_df = output_df.drop(['time'], axis = 1)
    return output_df    

# Process date in training dataset

df_train_date_processed = process_date(df_train_no_unnamed)
df_train_date_processed.head(2)

# Process date in training dataset

df_train_date_processed = process_date(df_train_no_unnamed)
df_train_date_processed.head(2)


# Process date in training dataset

df_train_date_processed = process_date(df_train_no_unnamed)
df_train_date_processed.head(2)


# Function to extract level number

def wind_deg_level(input_df):
    output_df = input_df.copy()
    for index, row in output_df.iterrows():
        wind_deg = row['Valencia_wind_deg']
        level = re.sub(r'\D', '', wind_deg)
        output_df.at[index, 'Valencia_wind_deg'] = level
    output_df['Valencia_wind_deg'] = pd.to_numeric(output_df['Valencia_wind_deg'])
    return output_df

# Update in the train set

df_train_Valencia_wind_updated = wind_deg_level(df_train_date_processed)
df_train_Valencia_wind_updated.head(2)

# Update in the test set

df_test_Valencia_wind_updated = wind_deg_level(df_test_date_processed)
df_test_Valencia_wind_updated.head(2)

# Comparing {city}_pressure columns

df_train_pressure = df_train_Valencia_wind_updated[['Seville_pressure', 'Barcelona_pressure', 'Bilbao_pressure', 'Valencia_pressure', 'Madrid_pressure']]
df_train_pressure.head()

# Function to extract level number

def pressure_level(input_df):
    output_df = input_df.copy()
    for index, row in output_df.iterrows():
        pressure = row['Seville_pressure']
        level = re.sub(r'\D', '', pressure)
        output_df.at[index, 'Seville_pressure'] = int(level)
    output_df['Seville_pressure'] = pd.to_numeric(output_df['Seville_pressure'])
    return output_df

    # Update in the train set

df_train_Seville_pressure_updated = pressure_level(df_train_Valencia_wind_updated)
df_train_Seville_pressure_updated[['Seville_pressure', 'Barcelona_pressure', 'Bilbao_pressure', 'Valencia_pressure', 'Madrid_pressure']].head(2)

# Update in the test set

df_test_Seville_pressure_updated = pressure_level(df_test_Valencia_wind_updated)
df_test_Seville_pressure_updated[['Seville_pressure', 'Barcelona_pressure', 'Bilbao_pressure', 'Valencia_pressure', 'Madrid_pressure']].head(2)

# Group Valencia_pressure missing values by month and year

df_train_valencia_pressure = df_train_Seville_pressure_updated.copy()
df_valencia_nulls = df_train_valencia_pressure.Valencia_pressure.isnull().groupby([df_train_valencia_pressure['year'],df_train_valencia_pressure['month']]).sum().astype(int).reset_index(name='nulls')
df_valencia_nulls

# Function to perform conditional impute

def fill_pressure_null(input_df):
    output_df = input_df.copy()
    for index, row in output_df.iterrows():
        if(pd.isnull(row['Valencia_pressure'])):
            conditions = list(output_df[['month', 'year']].iloc[index])
            filtered = output_df[(output_df['month'] == conditions[0]) & (output_df['year'] == conditions[1])]
            mode = list(filtered['Valencia_pressure'].mode())
            output_df.at[index, 'Valencia_pressure'] = mode[0]
    return output_df

# Fill in the training set

df_train_no_nulls = fill_pressure_null(df_train_Seville_pressure_updated)
df_train_no_nulls['Valencia_pressure'].head(5)

# Fill in the test set

df_test_no_nulls = fill_pressure_null(df_test_Seville_pressure_updated)
df_test_no_nulls['Valencia_pressure'].head(5)

# Train set confirmation

df_train_no_nulls.info()

# Test set confirmation

df_test_no_nulls.info()

# Perform prediction with model and preprocessed data.
prediction = model.predict(prep_data)
# Format as list for output standardisation.
return prediction[0].tolist()


