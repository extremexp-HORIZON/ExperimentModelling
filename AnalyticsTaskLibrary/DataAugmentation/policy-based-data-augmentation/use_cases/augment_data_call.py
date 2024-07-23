import requests
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the breast cancer dataset from scikit-learn
breast_cancer = load_breast_cancer()
#Feature names must not hafve white spaces, and must contain a label column
data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
data['label'] = breast_cancer.target
data = data.rename(columns=lambda x: x.replace(' ', '_'))
# Convert the dataframe toa list of dicts
dataset = data.to_dict(orient='records') 

# Define the payload for the POST request
payload = {
    "dataset": dataset,
    "method": "AUTO", #CTGAN,SMOTE, ADASYN, AUTO
    "label_column_name": "label",
    "n_samples": 50,
    "distance": "pairwise" #wassertein, pairwise
}
 
url = "http://127.0.0.1:8000/augment_data"

# Make the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    try:
        augmented_data_list = response.json()["augmented_data"]
        distance = response.json()["distances"]
        # Create a new DataFrame called augmented_data
        augmented_data = pd.DataFrame.from_records(augmented_data_list)
        print(f"Augmented Data: {augmented_data.shape}")
    except Exception as e:
        print(f"Error processing response JSON: {e}")
else:
    print(f"Error: {response.status_code}, {response.text}")

print(augmented_data["label"].value_counts())
print(distance)

