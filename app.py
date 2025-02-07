from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

def preprocess_data(df):
    # Handle missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    # Standardize numeric data
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categorical data
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, encoded_cats_df], axis=1)

    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file)
        processed_df = preprocess_data(df)
        return processed_df.to_csv(index=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)