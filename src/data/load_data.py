import pandas as pd
import os

def load_data(file_path:str)->pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise