# Bayesian ML Credit Scoring

This project evaluates credit default risk using a Naïve Bayes Gaussian classifier on the UCI Credit Card dataset.

## Project content

- `ProjetBayesianML.ipynb` - Notebook showing data analysis, model training, evaluation, and example prediction.
- `credit_scoring_interface.py` - Streamlit web interface to interactively score new client profiles.
- `UCI_Credit_Card.csv` - Dataset used to train the model.

## Features

- Data loading and preprocessing
- Model training with `GaussianNB`
- Model evaluation with confusion matrix, ROC AUC, and precision-recall
- Interactive client score interface with Streamlit

## Requirements

Use the provided `requirements.txt` to install dependencies.

```bash
pip install -r requirements.txt
```

## Run locally

From the project folder:

```bash
streamlit run "credit_scoring_interface.py"
```

Then open your browser at:

```bash
http://localhost:8501
```

## Prepare for GitHub deployment

1. Create a GitHub repository.
2. Add the project files and commit:

```bash
git init
 git add .
 git commit -m "Initial commit"
 git branch -M main
 git remote add origin <your-git-url>
 git push -u origin main
```

3. To deploy the Streamlit app on Streamlit Community Cloud:
   - Push the repo to GitHub.
   - Sign in to [Streamlit Cloud](https://streamlit.io/cloud).
   - Create a new app from your GitHub repository.
   - Set the main file to `credit_scoring_interface.py`.

## Notes

- Keep `UCI_Credit_Card.csv` in the same repository root as the script.
- If you want to improve production readiness, consider replacing the live model training with a saved model file and use `joblib` or `pickle` for loading.
- The app currently retrains the model at startup using the dataset.
