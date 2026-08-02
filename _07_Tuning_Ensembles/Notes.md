# Machine Learning Notes
### Model Tuning, Ensemble Learning, Unsupervised Learning

---

## Table of Contents

1. [Model Tuning](#1-model-tuning)
2. [Hyperparameters](#2-hyperparameters)
3. [Cross Validation](#3-cross-validation)
4. [Hyperparameter Tuning](#4-hyperparameter-tuning)
5. [Grid Search CV](#5-grid-search-cv)
6. [Random Search CV](#6-random-search-cv)
7. [Ensemble Learning](#7-ensemble-learning)
8. [Bagging (Bootstrap Aggregating)](#8-bagging-bootstrap-aggregating)
9. [Boosting](#9-boosting)
10. [AdaBoost (Adaptive Boosting)](#10-adaboost-adaptive-boosting)
11. [Gradient Boosting](#11-gradient-boosting)
12. [XGBoost (Extreme Gradient Boosting)](#12-xgboost-extreme-gradient-boosting)
13. [Unsupervised Learning](#13-unsupervised-learning)
14. [K-Means Clustering Algorithm](#14-k-means-clustering-algorithm)
15. [DBSCAN Algorithm (Density-Based Spatial Clustering)](#15-dbscan-algorithm-density-based-spatial-clustering)
16. [Dimensionality Reduction](#16-dimensionality-reduction)
17. [PCA (Principal Component Analysis)](#17-pca-principal-component-analysis)
18. [Random Forest Algorithm](#18-random-forest-algorithm)
19. [Quick Summary Table](#quick-summary-table)

---

## 1. Model Tuning

**Simple Definition:**
Model tuning means changing the settings of your model to make it work better. Just like tuning a guitar so it sounds good, we "tune" a model so it gives good predictions.

**Interview Explanation:**
"Model tuning is the process of adjusting a machine learning model's settings (called hyperparameters) so that it gives the best performance on unseen data. It's different from training, because training learns the model's internal values, but tuning decides the settings we give to the model before training starts."

**Formula:**
No single formula. It's a process: Train model → Check score → Change settings → Train again → Compare scores → Pick best settings.

**Example:**
You have a Decision Tree. You try `max_depth = 3`, then `max_depth = 5`, then `max_depth = 10`. You pick the depth that gives the best accuracy on test data. This is model tuning.

---

## 2. Hyperparameters

**Simple Definition:**
Hyperparameters are settings you choose **before** training a model. The model does not learn them itself — you set them.

**Interview Explanation:**
"Hyperparameters are configuration values set before training, like `k` in KNN or `max_depth` in Decision Tree. They control how the model learns. This is different from parameters (like weights in Linear Regression), which the model learns automatically from data."

**Formula:**
No math formula. Just examples:
- KNN → `n_neighbors`
- Decision Tree → `max_depth`, `min_samples_split`
- SVM → `C`, `kernel`
- Random Forest → `n_estimators`

**Example:**
In KNN, if you set `k=5`, the model will look at 5 nearest neighbors to make a prediction. `k=5` is a hyperparameter because you chose it, the model didn't learn it.

---

## 3. Cross Validation

**Simple Definition:**
Cross validation is a way to test your model on different parts of your data, so you get a fair idea of how good your model really is.

**Interview Explanation:**
"Cross validation splits the dataset into multiple parts (called folds). The model trains on some folds and tests on the remaining fold. This is repeated so every part of data gets used for testing once. It gives a more reliable accuracy score than a single train-test split, and helps detect overfitting."

**Formula (K-Fold Cross Validation Score):**
```
CV Score = (Score_1 + Score_2 + Score_3 + ... + Score_k) / k
```

**How the formula works:**
You split data into `k` equal parts (folds). You train on `k-1` folds and test on the remaining 1 fold. You repeat this `k` times, each time a different fold is used for testing. Then you take the average of all `k` scores. That average is your final, more trustworthy score.

**Example:**
You have 500 rows of data. You do 5-Fold Cross Validation (`k=5`):
- Fold 1 test → accuracy 80%
- Fold 2 test → accuracy 82%
- Fold 3 test → accuracy 78%
- Fold 4 test → accuracy 85%
- Fold 5 test → accuracy 81%

Final CV Score = (80+82+78+85+81)/5 = **81.2%**

---

## 4. Hyperparameter Tuning

**Simple Definition:**
Hyperparameter tuning means trying different hyperparameter values and picking the combination that gives the best result.

**Interview Explanation:**
"Hyperparameter tuning is the process of searching for the best hyperparameter values for a model. We usually combine it with cross validation, so that the chosen hyperparameters are not just lucky on one split of data, but perform well overall. Common techniques are Grid Search CV and Random Search CV."

**Formula:**
No fixed formula — it's a search process:
```
Best Params = argmax( CV_Score(params) ) for all params tried
```

**How it works:**
We try many combinations of hyperparameters. For each combination, we calculate the cross-validation score. Whichever combination gives the highest score is chosen as the "best hyperparameters."

**Example:**
For Decision Tree, we try:
- `max_depth=3, min_samples_split=2` → CV score 75%
- `max_depth=5, min_samples_split=4` → CV score 83%
- `max_depth=7, min_samples_split=2` → CV score 79%

We pick `max_depth=5, min_samples_split=4` because it has the highest score.

---

## 5. Grid Search CV

**Simple Definition:**
Grid Search CV tries **every possible combination** of hyperparameters you give it, and tells you which one is best.

**Interview Explanation:**
"GridSearchCV is a brute-force hyperparameter tuning method. You give it a dictionary of hyperparameters with possible values. It trains and evaluates the model using cross validation for every single combination, then returns the combination with the best score. It is accurate but can be slow if there are many combinations."

**Flow :** 
First, we choose a model and define its hyperparameters with different values. Then GridSearchCV creates all possible combinations of these hyperparameters. For each combination, it applies cross validation on the dataset by splitting it into folds and calculates the average accuracy. After testing all combinations, it selects the best hyperparameters based on the highest score and returns the best model.

**Formula (Total combinations tried):**
```
Total Fits = (values of param1) × (values of param2) × ... × k folds
```

**How it works:**
If you give 3 values for `max_depth` and 2 values for `min_samples_split`, Grid Search will try 3 × 2 = 6 combinations. With 5-fold cross validation, it actually trains the model 6 × 5 = 30 times, and picks the best.

**Example (Python-style):**
```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4]
}

grid = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)   # best combination
print(grid.best_score_)    # best CV score
```

---

## 6. Random Search CV

**Simple Definition:**
Random Search CV does not try every combination. It randomly picks a fixed number of combinations to test, which makes it faster than Grid Search.

**Interview Explanation:**
"RandomizedSearchCV picks random combinations of hyperparameters instead of testing all of them like GridSearchCV. This is much faster, especially when there are many hyperparameters with many possible values. It usually finds a good (not always the absolute best) combination in much less time."

**Formula (Total fits):**
```
Total Fits = n_iter × k folds
```
(`n_iter` = number of random combinations you want to try)

**How it works:**
Instead of checking all combinations, you tell it "just try 10 random combinations" (`n_iter=10`). It randomly picks 10 sets of hyperparameters, evaluates each with cross validation, and returns the best one found among those 10.

**Example (Python-style):**
```python
from sklearn.model_selection import RandomizedSearchCV

params = {
    'max_depth': [3, 5, 7, 9, 11],
    'min_samples_split': [2, 4, 6, 8]
}

random_search = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter=5, cv=5)
random_search.fit(X_train, y_train)

print(random_search.best_params_)
```

**Grid Search vs Random Search (quick compare):**
| Grid Search | Random Search |
|---|---|
| Tries all combinations | Tries random combinations |
| Slow with big data | Faster |
| Guaranteed best from list | May miss the true best |

---

## 7. Ensemble Learning

**Simple Definition:**
Ensemble learning means combining many small models together to make one strong model. It's like asking many people for their opinion instead of just one person, and then going with the majority answer.

**Interview Explanation:**
"Ensemble learning combines predictions from multiple models (called base learners or weak learners) to produce a better result than any single model alone. The idea is that different models make different mistakes, and combining them cancels out errors and improves accuracy. Main types are Bagging, Boosting, and Stacking."

**Formula (Simple Voting for Classification):**
```
Final Prediction = Mode(prediction_1, prediction_2, ..., prediction_n)
```

**Formula (Averaging for Regression):**
```
Final Prediction = (prediction_1 + prediction_2 + ... + prediction_n) / n
```

**How it works:**
Each model in the ensemble gives its own prediction. For classification, we take the answer most models agree on (majority vote). For regression, we take the average of all predictions.

**Example:**
3 models predict if an email is spam:
- Model 1 → Spam
- Model 2 → Spam
- Model 3 → Not Spam

Majority vote = **Spam** (2 out of 3 said spam).

---

## Stacking (Stacked Generalization)

**Simple Definition:**
Stacking is a way to combine predictions from several different models by feeding those predictions into another model, which then makes the final decision.

**Interview Explanation:**
"Stacking trains multiple different base models (like KNN, SVM, Decision Tree) on the same data. Instead of just averaging or voting on their outputs, we take their predictions and use them as new input features for another model, called the meta-model. The meta-model learns how to best combine the base models' predictions to make the final prediction. This usually performs better than any single model alone because it learns from the strengths and weaknesses of each base model."

**Formula (Meta-Model Prediction):**
```
Final Prediction = Meta_Model( Base_Model_1_pred, Base_Model_2_pred, ..., Base_Model_n_pred )
```

**How the formula works:**
You train `n` different base models (Level-0 models) on the training data. Each base model makes a prediction. These predictions become the new features (a new dataset) instead of the original raw features. You then train a meta-model (Level-1 model) on this new dataset of predictions, so it learns which base model to trust more in which situation. To avoid data leakage, the base model predictions used to train the meta-model are usually generated using cross validation (out-of-fold predictions), not predictions on the same data the base models were trained on.

**Example:**
You have a dataset for predicting if a customer will churn. You train 3 base models:
- KNN → predicts probability 0.70
- Decision Tree → predicts probability 0.60
- SVM → predicts probability 0.65

These 3 outputs become the new input row: `[0.70, 0.60, 0.65]`

This new row is fed into the meta-model (say, Logistic Regression), which learns from many such rows across the dataset and outputs the final prediction:

Final Prediction = Meta_Model([0.70, 0.60, 0.65]) → **0.68 (Churn = Yes)**

**Base Models vs Meta-Model:**

| Term | Role |
|---|---|
| Level-0 Models (Base Models) | Different algorithms trained directly on original data |
| Level-1 Model (Meta-Model) | Trained on the predictions of base models, not raw data |
| Out-of-Fold Predictions | Predictions made using CV, so meta-model doesn't see "cheated" results |

**Code Example (scikit-learn):**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define base models (Level-0)
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=4)),
    ('svm', SVC(probability=True))
]

# Define meta-model (Level-1)
meta_model = LogisticRegression()

# Build stacking classifier
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # uses cross validation internally to avoid data leakage
)

stack_model.fit(X_train, y_train)
predictions = stack_model.predict(X_test)
```

**Stacking vs Bagging vs Boosting (quick comparison):**

| Method | How base models are trained | How outputs are combined |
|---|---|---|
| Bagging | Parallel, on random data subsets | Simple average / majority vote |
| Boosting | Sequential, each model fixes previous errors | Weighted sum |
| Stacking | Parallel, different algorithm types | Learned by a meta-model |

**Key Point:**
Stacking works best when the base models are diverse (different algorithm types making different kinds of mistakes), because the meta-model has more varied information to learn from.

---

## 8. Bagging (Bootstrap Aggregating)

**Simple Definition:**
Bagging trains many copies of the same model on different random samples of data, then combines their answers. It reduces overfitting.

**Interview Explanation:**
"Bagging stands for Bootstrap Aggregating. We create multiple random subsets of the training data (with replacement, called bootstrapping). We train one model on each subset, in parallel and independently. Finally, we combine their predictions using voting (classification) or averaging (regression). Random Forest is the most famous example of bagging."

**Formula:**
```
Final Prediction = (1/n) × Σ (prediction_i)     [for regression]
Final Prediction = Mode(prediction_1, ..., prediction_n)   [for classification]
```

**How it works:**
Each model is trained on a random sample of data picked **with replacement** (meaning the same row can be picked more than once). Because each model sees slightly different data, they make different mistakes. Averaging or voting reduces the overall error.

**Example:**
You have 1000 rows. You create 5 random samples (each with 1000 rows, picked with replacement) and train 5 Decision Trees, one on each sample. Combine their votes = Bagging.

---

## 9. Boosting

**Simple Definition:**
Boosting trains models **one after another**, and each new model tries to fix the mistakes of the previous model.

**Interview Explanation:**
"Boosting is a sequential ensemble technique. Unlike bagging, models are not trained independently — each new model focuses more on the data points that previous models got wrong. The final prediction is a weighted combination of all models. Boosting usually gives higher accuracy than bagging but can overfit if not controlled. Examples: AdaBoost, Gradient Boosting, XGBoost."

**Formula (General Boosting idea):**
```
Final Model = w1×Model_1 + w2×Model_2 + ... + wn×Model_n
```
(`w` = weight given to each model based on how good it performed)

**How it works:**
Model 1 is trained normally. Points it got wrong are given more "importance" (higher weight). Model 2 is trained focusing on those hard points. This continues for many rounds. At the end, all models vote, but better-performing models get more say (higher weight).

**Example:**
Model 1 wrongly predicts 10 out of 100 emails. Model 2 is trained giving extra attention to those 10 wrong emails. Model 3 focuses on whatever Model 2 still got wrong. This chain of "fixing mistakes" continues.

---

## 10. AdaBoost (Adaptive Boosting)

**Simple Definition:**
AdaBoost is a boosting method where wrong predictions are given more weight, so the next model tries harder on them.

**Interview Explanation:**
"AdaBoost works by assigning weights to each training sample. Initially, all samples have equal weight. After each weak learner (usually a small decision tree called a stump), the weight of misclassified samples is increased, so the next learner pays more attention to them. Each weak learner also gets a weight based on its accuracy. The final prediction is a weighted vote of all weak learners."

**Formula (Weight update for a data point):**
```
new_weight = old_weight × e^(α × error)
```

**Formula (Model's say / weight - alpha):**
```
α = 0.5 × ln( (1 - total_error) / total_error )
```

**How the formula works:**
`α` (alpha) tells us how much "say" a weak model gets in the final vote. If a model has low error, `α` is high (it gets more say). If a model has high error (close to 50%, like random guessing), `α` is close to 0 (barely trusted). After finding `α`, we increase the weight of the points that were predicted wrong, so the next model pays more attention to them.

**Example:**
Weak learner 1 makes some mistakes. Its error rate is 20% (0.2).
```
α = 0.5 × ln((1-0.2)/0.2) = 0.5 × ln(4) ≈ 0.69
```
This model gets a decent "say" (0.69) in the final vote. The wrongly-classified points get their weight increased before training the next learner.

---

## 11. Gradient Boosting

**Simple Definition:**
Gradient Boosting builds models one by one, where each new model tries to predict the **error (residual)** of the previous model, not the original target.

**Interview Explanation:**
"Gradient Boosting is a boosting technique that builds trees sequentially. Instead of reweighting data points like AdaBoost, each new tree is trained to predict the residual errors (the difference between actual and predicted values) of the combined previous trees. It uses gradient descent to minimize a loss function. This makes it powerful but also more prone to overfitting if not tuned properly."

**Formula (Residual):**
```
Residual = Actual Value - Predicted Value
```

**Formula (Updated Prediction):**
```
New Prediction = Previous Prediction + (learning_rate × new_tree_prediction)
```

**How the formula works:**
First, we make a simple prediction (like the average). Then we calculate the residual (how far off we were). A new small tree is trained to predict that residual. We add this new tree's prediction to our old prediction, but scaled down by a `learning_rate` (a small number like 0.1) so learning happens slowly and doesn't overfit. This repeats for many rounds.

**Example:**
Actual house price = 300,000
Model's first guess (average) = 250,000
Residual = 300,000 - 250,000 = 50,000

Next tree tries to predict this 50,000 gap. If learning_rate = 0.1 and the tree predicts 45,000:
```
New Prediction = 250,000 + (0.1 × 45,000) = 254,500
```
Slowly, over many trees, the prediction gets closer to 300,000.

---

## 12. XGBoost (Extreme Gradient Boosting)

**Simple Definition:**
XGBoost is an improved, faster, and more powerful version of Gradient Boosting. It is one of the most popular algorithms used in competitions and real jobs.

**Interview Explanation:**
"XGBoost stands for Extreme Gradient Boosting. It works on the same idea as Gradient Boosting — building trees sequentially to fix previous errors — but it adds regularization (to prevent overfitting), handles missing values automatically, supports parallel processing (faster training), and uses second-order gradients (more accurate updates) for better performance."

**Formula (Objective Function XGBoost minimizes):**
```
Objective = Loss(actual, predicted) + Regularization(tree complexity)
```

**How the formula works:**
XGBoost doesn't just try to reduce prediction error (Loss); it also punishes trees that are too complex (too many leaves, too deep) using the Regularization term. This keeps the model simple and prevents overfitting, while still learning well from data.

**Example (Python-style):**
```python
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
XGBoost is often used because it's fast and gives high accuracy with tabular data (like CSV, Excel data), which is very common in real jobs.

---

## 13. Unsupervised Learning

**Simple Definition:**
Unsupervised learning is when the model learns from data that has **no labels/answers**. The model has to find patterns by itself.

**Interview Explanation:**
"In unsupervised learning, we only give the model input data (X) without any target/output (y). The model's job is to find hidden patterns, groupings, or structure in the data on its own. Common tasks are clustering (grouping similar data) and dimensionality reduction (simplifying data). This is different from supervised learning, where we give the model the correct answers to learn from."

**Formula:**
No single formula — depends on the algorithm used (K-Means, DBSCAN, PCA, etc. — each explained below).

**Example:**
You give a model 10,000 customer records (age, spending, income) with **no labels** saying which customer is "high value" or "low value". The model groups similar customers together on its own. This is unsupervised learning.

---

## 14. K-Means Clustering Algorithm

**Simple Definition:**
K-Means groups data into `k` clusters (groups) based on how close data points are to each other.

**Interview Explanation:**
"K-Means is a clustering algorithm that divides data into K clusters. It works by placing K random center points (centroids), assigning each data point to the nearest centroid, then recalculating the centroid as the average of its assigned points. This repeats until the centroids stop moving. The goal is to minimize the distance between points and their cluster's centroid."

**Formula (Euclidean Distance — used to assign points):**
```
distance = √[ (x2-x1)² + (y2-y1)² ]
```

**Formula (Within-Cluster Sum of Squares — WCSS, what K-Means minimizes):**
```
WCSS = Σ Σ (distance between point and its cluster centroid)²
```

**How the formula works:**
K-Means picks `k` random centroids at first. Every data point is measured by its distance to each centroid (using Euclidean distance formula). Each point joins the nearest centroid's group. Then the centroid moves to the average position of all points in its group. This repeats until centroids stop moving much. WCSS tells us how tight/compact the clusters are — lower WCSS is better.

**Example:**
You have customer data with "Annual Income" and "Spending Score." You set `k=3`. K-Means groups customers into 3 clusters:
- Cluster 1: Low income, low spending
- Cluster 2: High income, high spending
- Cluster 3: High income, low spending

**How to choose k:** Use the "Elbow Method" — plot WCSS for different k values, and pick the k where the drop in WCSS starts to flatten (looks like an elbow).

---

## 15. DBSCAN Algorithm (Density-Based Spatial Clustering)

**Simple Definition:**
DBSCAN groups points that are close together (dense areas) into clusters, and marks points that are far from everyone else as "noise" (outliers). Unlike K-Means, you don't need to say how many clusters you want.

**Interview Explanation:**
"DBSCAN is a density-based clustering algorithm. It groups together points that are closely packed, and marks points in low-density regions as outliers/noise. It uses two parameters: `eps` (the radius to check around a point) and `min_samples` (minimum number of points needed to form a dense region). Unlike K-Means, DBSCAN can find clusters of any shape and doesn't require specifying the number of clusters beforehand."

**Formula (Core point condition):**
```
A point is a Core Point if:
  (number of points within distance 'eps' of it) ≥ min_samples
```

**How it works:**
For every point, DBSCAN checks how many other points are within a circle of radius `eps` around it. If that count is ≥ `min_samples`, it's called a **Core Point** (center of a dense area). Points near a core point but not dense enough themselves are **Border Points**. Points that are alone, with too few neighbors, are marked as **Noise/Outliers**. Clusters are formed by connecting nearby core points together.

**Example:**
Imagine dots on a map showing where people are standing.
- `eps = 2 meters`, `min_samples = 5`
- If a person has 5 or more other people within 2 meters → they are a Core Point, part of a crowd (cluster).
- If someone is standing alone far away → Noise (outlier), not part of any group.

This is useful for finding oddly-shaped clusters, like fraud detection, where fraud points are usually "outliers" that DBSCAN naturally separates.

---

## 16. Dimensionality Reduction

**Simple Definition:**
Dimensionality reduction means reducing the number of columns (features) in your data, while still keeping the important information.

**Interview Explanation:**
"Dimensionality reduction reduces the number of input features (dimensions) in a dataset while trying to preserve as much important information as possible. It helps in reducing overfitting, speeding up training, and helps visualize high-dimensional data in 2D or 3D. The most common technique is PCA (Principal Component Analysis)."

**Formula:**
No single fixed formula — depends on the method (PCA explained next).

**Example:**
You have a dataset with 100 columns describing a house (size, rooms, location, age, paint color, etc). Many of these columns might be similar or unimportant. Dimensionality reduction can shrink this to, say, 10 important columns that still explain most of the pattern in the data.

---

## 17. PCA (Principal Component Analysis)

**Simple Definition:**
PCA is a technique that takes many columns of data and combines them into fewer new columns (called Principal Components), while keeping as much important information as possible.

**Interview Explanation:**
"PCA is a dimensionality reduction technique that transforms the original correlated features into a smaller set of uncorrelated features called principal components. These components are ordered by how much variance (information) they capture from the original data. The first principal component captures the most variance, the second captures the next most, and so on. PCA is often used before training models to reduce noise, remove multicollinearity, and speed up computation."

**Formula (Variance — what each component tries to capture):**
```
Variance = Σ (xi - mean)² / n
```

**Formula (Covariance Matrix — relationship between features):**
```
Cov(X,Y) = Σ (xi - mean_x)(yi - mean_y) / n
```

**How the formula works:**
PCA first calculates the covariance matrix, which shows how each feature relates to the others (do they increase/decrease together?). From this matrix, it calculates special directions called **eigenvectors** (the new axes) and **eigenvalues** (how much variance/information is along each axis). It picks the top eigenvectors with the highest eigenvalues as the new Principal Components, since they hold the most information.

**Example:**
You have a dataset with "Height in cm" and "Height in inches" — these two columns basically carry the same information (they are correlated). PCA can combine them into a single new component that still represents "height," reducing 2 columns into 1 without losing much information.

In practice with many features:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)   # reduce to 2 components
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_)  # how much info each component keeps
```

---

## 18. Random Forest Algorithm

**Simple Definition:**
Random Forest is a group (forest) of many Decision Trees. Each tree gives a vote, and the majority vote becomes the final answer. It's a type of Bagging.

**Interview Explanation:**
"Random Forest is an ensemble learning algorithm based on Bagging. It builds multiple decision trees, each trained on a random subset of data (bootstrap sampling) and a random subset of features. For classification, the final output is the majority vote of all trees. For regression, it's the average of all tree outputs. Random Forest reduces overfitting compared to a single decision tree because errors from individual trees average out."

**Formula (Classification — Majority Vote):**
```
Final Prediction = Mode(tree_1, tree_2, ..., tree_n)
```

**Formula (Regression — Average):**
```
Final Prediction = (tree_1 + tree_2 + ... + tree_n) / n
```

**How the formula works:**
Each tree is trained on a different random sample of the data (with replacement) AND a random subset of features (this is what makes it "random," and different from plain bagging). Because each tree sees different data and different features, they make different mistakes. When you combine (vote/average) all trees together, individual mistakes cancel out, giving a stronger, more stable final prediction.

**Example:**
You build a Random Forest with 5 trees to predict if a loan should be approved:
- Tree 1 → Approve
- Tree 2 → Approve
- Tree 3 → Reject
- Tree 4 → Approve
- Tree 5 → Reject

Majority vote = **Approve** (3 out of 5 trees said approve).

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Quick Summary Table

| Topic | Type | Key Idea |
|---|---|---|
| Cross Validation | Evaluation | Test model on multiple data splits for a fair score |
| Grid Search CV | Tuning | Try every combination of hyperparameters |
| Random Search CV | Tuning | Try random combinations, faster |
| Bagging | Ensemble | Many models trained in parallel on random samples, then vote/average |
| Boosting | Ensemble | Models trained one after another, fixing previous mistakes |
| AdaBoost | Boosting | Increases weight of wrongly predicted points |
| Gradient Boosting | Boosting | Each new tree predicts the residual error |
| XGBoost | Boosting | Faster, regularized version of Gradient Boosting |
| Random Forest | Bagging | Many decision trees, random data + random features, majority vote |
| K-Means | Clustering | Groups data into k clusters based on distance to centroid |
| DBSCAN | Clustering | Groups based on density, finds outliers, no need to set k |
| PCA | Dim. Reduction | Combines correlated columns into fewer important components |

---