# Steps Performed in EDA Insurance Notebook

This document outlines the sequential steps and specific functions/algorithms used in the data analysis pipeline of `eda_insurance.ipynb`.

---

## 1. Load Data
*   **Function/Library:** `pd.read_csv()` (Pandas)
*   **Action:** Loads the raw dataset `insurance.csv` into a DataFrame.

## 2. Exploratory Data Analysis (EDA) & Visualization
*   **Statistical Functions:** `df.shape`, `df.describe()`, and `df.isnull().sum()` to examine the dataset structure, statistics, and check for missing values.
*   **Plotting Functions:** 
    *   `mlt.hist()` (Matplotlib) to check numeric feature distributions (age, bmi, charges) and counts for categorical features (sex, smoker).
    *   `mlt.boxplot()` (Matplotlib) to scan for outliers in numerical features.

## 3. Data Cleaning & Encoding
*   **Deduplication:** `df.drop_duplicates()` to remove duplicate rows.
*   **Binary Mapping:** `.map({"male": 0, "female": 1})` and `.map({"yes": 1, "no": 0})` to map binary text features (`sex`, `smoker`) into numerical values (`is_femail`, `is_smoker`).
*   **One-Hot Encoding:** `pd.get_dummies()` to convert multi-class categorical features (`region`) into binary dummy columns.

## 4. Feature Engineering & Scaling
*   **Binning:** `pd.cut()` to segment numerical `bmi` values into discrete groups (underweight, normal, overweight, obese).
*   **One-Hot Encoding & Casting:** `pd.get_dummies(..., dtype=bool)` followed by `df.astype(int)` to convert the new `clean_bmi` groups into integer flags.
*   **Dropping Column:** `df.drop(columns=['bmi'])` to remove the redundant continuous column.
*   **Feature Scaling:** `StandardScaler().fit_transform()` (scikit-learn) to standardize continuous features (`age`, `children`) to have a mean of 0 and a standard deviation of 1.

## 5. Feature Selection
*   **Method 1 (SelectKBest):** `SelectKBest(score_func=f_classif, k=1)` (scikit-learn) to select features using ANOVA F-value test.
*   **Method 2 (Pearson Correlation):** `pearsonr()` (SciPy) to calculate linear correlation coefficients between features and the target variable (`charges`).
*   **Method 3 (Chi-Square Test):** `pd.qcut()` to bin continuous `charges` into equal-sized quartiles, followed by `chi2_contingency()` (SciPy) on `pd.crosstab()` to identify statistically significant categorical dependencies.
