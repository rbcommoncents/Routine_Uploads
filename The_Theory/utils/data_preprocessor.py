import pandas as pd 

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def preprocess_data(data, is_training=True):
    if is_training and 'label' in data.columns:
        features = data.drop('label', axis=1)
        labels = data['label']
        normalized_features = normalize_data(features)
        return normalized_features, labels
    else:
        normalized_features = normalize_data(data)
        return normalized_features