import pickle, os, json, random
from sklearn.metrics import f1_score
import joblib, glob, sys
import argparse
import pandas as pd
import urllib.request
import zipfile

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
        model = joblib.load(f'models/{model_version}.joblib')
    except:
        raise ValueError('Failed to catching the latest model')
        
    try:
        # Load spam dataset
        dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        data_file = 'SMSSpamCollection'
        
        if not os.path.exists(data_file):
            # Download the dataset if it doesn't exist
            zip_file = 'smsspamcollection.zip'
            if not os.path.exists(zip_file):
                print("Downloading SMS Spam Collection dataset...")
                urllib.request.urlretrieve(dataset_url, zip_file)
            
            # Extract the data file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
        
        # Load the dataset
        df = pd.read_csv(data_file, sep='\t', header=None, names=['label', 'message'])
        
        # Convert labels to binary (spam=1, ham=0)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Sample data for evaluation
        n_samples = random.randint(1000, min(5572, len(df)))
        df = df.sample(n=n_samples, random_state=0).reset_index(drop=True)
        
        # Split features and target
        X_text = df['message'].values
        y = df['label'].values
        
        # Load vectorizer
        if os.path.exists('data/vectorizer.pickle'):
            with open('data/vectorizer.pickle', 'rb') as vec_file:
                vectorizer = pickle.load(vec_file)
        else:
            raise ValueError('Vectorizer not found. Please train the model first.')
        
        # Vectorize text data
        X = vectorizer.transform(X_text).toarray()
    except Exception as e:
        raise ValueError(f'Failed to load the data: {str(e)}')
    
    y_predict = model.predict(X)
    metrics = {"F1_Score":f1_score(y, y_predict)}
    
    # Save metrics to a JSON file
    # Create metrics directory if it doesn't exist
    if not os.path.exists('metrics'): 
        os.makedirs('metrics')
    
    metrics_filename = f'metrics/{timestamp}_metrics.json'
    with open(metrics_filename, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    print(f"Metrics saved to: {metrics_filename}")
               
    
