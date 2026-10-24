# Linear Regression with Gradient Descent from Scratch

> 📚 **Course lab** – Machine Learning / Linear Regression exercises (M04W01)

Multiple linear regression implemented **from scratch with NumPy** (no scikit-learn) on the classic **Advertising** dataset (200 samples: `TV`, `Radio`, `Newspaper` → `Sales`).

## What is implemented

| Notebook | Content |
|---|---|
| `Question_1_2_3_Advertising_Stochastic_Problem.ipynb` | Data loading (`prepare_data`), prediction, **MSE and MAE loss**, gradients for each weight and bias, **stochastic gradient descent**, and an n-sample (mini-batch) variant |
| `Question4_Adversiting_Problem.ipynb` | Full training loop with a fixed initialization, loss tracking per epoch |
| `Question5_Adversiting_Problem.ipynb` | Same model with **min-max scaling** of the features before training |

## Getting started

```bash
git clone https://github.com/HuongPQ-w/linear-regression-sgd-advertising.git
cd linear-regression-sgd-advertising/notebooks
pip install -r ../requirements.txt
jupyter notebook
```

Open a notebook and run the cells in order. `advertising.csv` is in the same folder.

## Tech stack

Python · NumPy · Matplotlib · Jupyter
