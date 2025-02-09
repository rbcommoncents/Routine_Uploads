import pandas as pd 

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def preprocess_data(data):
    features = data.drop('label', axis=1)
    labels = data['label']
    normalized_features = normalize_data(features)
    return normalized_features, labels