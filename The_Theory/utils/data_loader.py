import pandas as pd 

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FilenotFoundError:
        print(f"File {file_path} not found")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def split_dataset(data, test_size=0.2):
    if data is not None:
        train_data = data.sample(frac=1 - test_size, random_state=42)
        test_data = data.drop(train_data.index)
        return train_data, test_data

    else:
        print("No data provided to split.")
        return None, None
        
