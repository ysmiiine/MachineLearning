
# main.py
from scripts.paths import RAW_DATA_PATH, CLEANSED_DATA_PATH, PREPROCESSED_DATA_PATH, RAW_TEXT_PATH, VOTING_MODEL_PATH,TRANSFORMER_MODEL_PATH
from scripts.preprocessing import preprocess_data
from scripts.training import train_model
from scripts.evaluation import evaluate_model
from scripts.evaluation2 import evaluate_model2
from scripts.training2 import train_transformer_model

# Preprocess data
#preprocess_data(RAW_DATA_PATH, CLEANSED_DATA_PATH, PREPROCESSED_DATA_PATH, RAW_TEXT_PATH)

#Train model
#model = train_model(PREPROCESSED_DATA_PATH, VOTING_MODEL_PATH)


# Evaluate model
evaluate_model(VOTING_MODEL_PATH, PREPROCESSED_DATA_PATH)

