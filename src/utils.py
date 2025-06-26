"""
Utility functions for student marks prediction.
"""

import joblib

def save_model(model, path):
    """
    Saves the trained model to disk.
    """
    joblib.dump(model, path)

def load_model(path):
    """
    Loads a trained model from disk.
    """
    return joblib.load(path)