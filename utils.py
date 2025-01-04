
# utils.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, vectorizer):
    # Assuming LightGBM is part of the VotingClassifier ensemble
    lgb_model = model.estimators_[2]
    importance = lgb_model.feature_importances_
    feature_names = vectorizer.get_feature_names_out()

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Important Features")
    plt.gca().invert_yaxis()
    plt.show()

