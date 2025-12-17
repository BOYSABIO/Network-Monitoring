# Evaluate the model
import joblib
from src.data_load.data_loader import load_data
from src.preprocess.preprocess import preprocess
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix

def evaluate(model):
    model = model

    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    logging.info("Iterations used per class:", model.n_iter_)
    logging.info("Max iterations allowed:", model.max_iter)
    logging.info(f"Accuracy: {accuracy:.2f}")

    logging.info("Classification Report")
    logging.info(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    logging.info("Confusion Matrix:")
    logging.info(cm)
    logging.info({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Threshold': thresholds})

def main():
    df = load_data()
    preprocessed = preprocess(df)

if __name__ == "__main__":
    main()