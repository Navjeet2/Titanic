# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster, using passenger details such as class, sex, age, family size, and fare. It covers data cleaning, encoding, and a comparison of five classification models, with cross-validation on the best performers.

## Overview

The goal is a binary classification: `survived` (1) or did not survive (0). I trained five classifiers on the same train/test split, compared them using accuracy, confusion matrices, and classification reports, and then checked the stability of the SVM with 5-fold cross-validation.

## Dataset

- **Source:** the Titanic dataset built into Seaborn (`sns.load_dataset("titanic")`), so no separate CSV is needed
- **Size:** 891 passengers, 15 columns originally (889 rows and 8 features after cleaning)
- **Target:** `survived`
- **Features used:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `alone`

## Workflow

### 1. Data Cleaning
- Dropped columns that were redundant or mostly empty: `deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`.
- Filled the missing `age` values (177 of 891) with the mean age.
- Dropped the 2 rows with a missing `embarked` value.

### 2. Encoding
- Label-encoded `sex` and `embarked` with `LabelEncoder`.
- Converted boolean and numeric columns to integers.

### 3. Train/Test Split
- 80/20 split (`random_state=42`), giving 711 training rows and 178 test rows.
- Standardized features with `StandardScaler` for the scale-sensitive models (KNN, SVM, Decision Tree).

### 4. Modeling
Trained and evaluated five classifiers with scikit-learn:

| Model | Test Accuracy |
|---|---|
| **Logistic Regression** | **0.8034** |
| **SVM (linear kernel)** | **0.8034** |
| KNN (k = 5) | 0.7753 |
| Naive Bayes | 0.7753 |
| Decision Tree | 0.7697 |

For the SVM, the confusion matrix on the test set was `[[91, 18], [17, 52]]`, with precision, recall, and F1 of about 0.75 for the survivor class.

### 5. Cross-Validation
- Ran 5-fold cross-validation on the SVM to check that the result does not depend on one particular split.
- Fold accuracies: 0.803, 0.809, 0.787, 0.753, 0.785
- **Mean CV accuracy: 0.787**

## Key Takeaways
- Logistic Regression and the linear SVM performed best, at about 80% test accuracy.
- The cross-validated accuracy (about 79%) is slightly below the single-split result, which is a more honest estimate of how the model would perform on new data.
- Simple models matched or beat the Decision Tree, which tends to overfit on small datasets like this one.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open `Titanic.ipynb` and run all cells. Seaborn downloads the dataset automatically, so an internet connection is needed the first time.

## Possible Improvements
- Add feature engineering, such as family size, a title extracted from the name, and age groups.
- Use one-hot encoding for `embarked` instead of label encoding, since it has no natural order.
- Fill missing ages using the median age per passenger class or title instead of the overall mean.
- Tune hyperparameters with `GridSearchCV` and try ensemble models such as Random Forest and Gradient Boosting.
- Cross-validate every model, not only the SVM.


