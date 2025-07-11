# Student Marks Prediction (Streamlit)

## Project Overview
Predicts student exam scores based on lifestyle and academic habits using machine learning.

## Live Demo
Try the app here: [Student Marks Prediction Live Demo](https://studentmarksprediction-d5f4xp6ekmdxp2zmxdx2hc.streamlit.app)

## Structure
- data/: Raw dataset
- models/: Saved ML models
- src/: Source code for preprocessing, training, and prediction
- streamlit_app.py: Streamlit web app
- requirements.txt: Python dependencies

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the model:
   ```
   python -m src.train
   ```
3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

