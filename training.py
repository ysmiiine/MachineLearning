import pickle
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from scipy.sparse import issparse

def train_model(preprocessed_data_path, model_path, chunk_size=1000):
    # Load preprocessed data
    with open(preprocessed_data_path, 'rb') as f:
        X_tfidf, y, _ = pickle.load(f)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Initialize models
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    # Voting Classifier
    voting_model = VotingClassifier(
        estimators=[
            ('log_reg', log_reg),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft'
    )

    # Train with progress bar
    total_samples = X_train.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  # Shuffle data

    with tqdm(total=total_samples, desc="Training Progress", leave=True) as pbar:
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_indices = indices[start_idx:end_idx]

            # Properly index sparse matrix or numpy array
            if issparse(X_train):
                X_chunk = X_train[chunk_indices].toarray()  # Convert sparse matrix to dense
            else:
                X_chunk = X_train[chunk_indices]

            # Use .iloc for pandas Series
            y_chunk = y_train.iloc[chunk_indices]

            # Train the model on the current chunk
            voting_model.fit(X_chunk, y_chunk)

            # Update the progress bar
            pbar.update(end_idx - start_idx)

    # Save the final model
    with open(model_path, 'wb') as f:
        pickle.dump(voting_model, f)

    print(f"Model saved to {model_path}")

    # Return the final model
    return voting_model
