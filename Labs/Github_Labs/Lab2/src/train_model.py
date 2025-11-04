import mlflow, datetime, os, pickle, random
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import urllib.request
import zipfile
import argparse

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Load spam dataset
    # Using SMS Spam Collection dataset
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
    
    # Sample data if needed for variation
    n_samples = random.randint(1000, min(5572, len(df)))  # Dataset has 5572 samples
    df = df.sample(n=n_samples, random_state=0).reset_index(drop=True)
    
    # Split features and target
    X_text = df['message'].values
    y = df['label'].values
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(X_text).toarray()
    # Save data and vectorizer
    if not os.path.exists('data'): 
        os.makedirs('data/')
    
    with open('data/data.pickle', 'wb') as data:
        pickle.dump(X, data)
        
    with open('data/target.pickle', 'wb') as data:
        pickle.dump(y, data)
    
    # Save vectorizer for evaluation
    with open('data/vectorizer.pickle', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)  
            
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "SMS Spam Collection"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        
        params = {
                    "dataset_name": dataset_name,
                    "number of datapoint": X.shape[0],
                    "number of dimensions": X.shape[1],
                    "model_type": "SVC",
                    "kernel": "rbf",
                    "max_features": 5000}
        
        mlflow.log_params(params)
            
        
        svm = SVC(kernel='rbf', random_state=0, probability=True)
        svm.fit(X_train, y_train)
        
        y_predict = svm.predict(X_test)
        mlflow.log_metrics({'Accuracy': accuracy_score(y_test, y_predict),
                            'F1 Score': f1_score(y_test, y_predict)})
        
        if not os.path.exists('models/'): 
            # then create it.
            os.makedirs("models/")
            
        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_dt_model.joblib'
        dump(svm, model_filename)
                    

