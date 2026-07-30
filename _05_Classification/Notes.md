# Machine Learning Classification

---

## 1. What is Classification?

**Definition (Simple):** Classification is a type of Machine Learning problem where we teach a computer to put things into **categories (classes)**, instead of predicting a number.

**Example:** 
- Is this email **Spam** or **Not Spam**?
- Is this tumor **Cancer** or **Not Cancer**?
- Is this animal a **Cat**, **Dog**, or **Bird**?

So basically, the output (answer) is always a **label/category**, not a continuous number like price or temperature.

### Important terms you need to know first:
- **Feature(s):** The input information we give to the model (e.g., email length, number of words like "free", "win").
- **Label / Target:** The correct answer/category we want to predict (e.g., Spam or Not Spam).
- **Model:** The "trained brain" that learns patterns from data and makes predictions.
- **Training Data:** Data we already know the answers for, used to teach the model.
- **Testing Data:** New/unseen data used to check how good the model is.

---

## 2. Types of Classification

### a) Binary Classification
Only **2 possible classes**.
- Example: Spam / Not Spam, Yes / No, Disease / No Disease

### b) Multi-Class Classification
**More than 2 classes**, but each data point belongs to **only one** class.
- Example: Classifying fruits into Apple, Banana, or Mango.

### c) Multi-Label Classification
One data point can belong to **more than one class at the same time**.
- Example: A movie can be labeled both "Action" and "Comedy" at the same time.

### d) Imbalanced Classification
When one class has **way more examples** than another class.
- Example: In fraud detection, 99% transactions are "Not Fraud" and only 1% are "Fraud". This makes the model biased, so we need special techniques to fix it.

---

## 3. Logistic Regression

**Definition (Simple):** Logistic Regression is a classification algorithm (even though the name has "Regression" in it) used to predict a **category**, usually **Yes/No or 0/1**.

It works by calculating the **probability** that something belongs to a certain class. If probability > 0.5 → class 1 (Yes), if probability < 0.5 → class 0 (No).

### How it works (simple explanation):
1. It takes your input features (like age, salary, etc.)
2. It calculates a weighted sum of these features (just like linear regression does).
3. Then it passes that sum through a special function called the **Sigmoid Function**.
4. The Sigmoid function squishes any number into a value **between 0 and 1** — this becomes our probability.

### Term: Sigmoid Function
A mathematical function shaped like an "S" curve. No matter how big or small the input number is, it always outputs a value between **0 and 1**. This is why Logistic Regression can give us probabilities.

### Term: Decision Boundary
This is the line (or curve) that separates one class from another. Anything on one side is class 0, and the other side is class 1.

---

## 4. Problem if we use Linear Regression for Classification

Many beginners ask: *"Why can't we just use Linear Regression to classify things?"*

Here are the problems we face:

1. **Output is not limited between 0 and 1**
   - Linear regression can give values like -50 or +200, but a probability must be between 0 and 1. This doesn't make sense for classification.

2. **Sensitive to Outliers**
   - If one data point is very extreme (far away), it can shift the whole line and mess up the boundary between classes.

3. **No Proper Decision Boundary**
   - Linear regression tries to fit a straight line through data points, but classification needs a clear boundary that separates classes properly — a straight line doesn't do this well, especially when data isn't linearly separated.

4. **Wrong Predictions Near Edges**
   - Since linear regression doesn't "squish" values, predictions can go beyond logical limits, making the model unreliable for yes/no type decisions.

**In short:** Linear Regression is made for predicting continuous numbers (like price, salary), not categories. That's why we use Logistic Regression instead — because of the Sigmoid function that keeps outputs between 0 and 1.

---

## 5. Logistic Regression vs Linear Regression

| Feature | Linear Regression | Logistic Regression |
|----------|-------------------|----------------------|
| Used for | Predicting continuous values (numbers) | Predicting categories (classes) |
| Output Range | Any number (-∞ to +∞) | Between 0 and 1 (probability) |
| Example | Predicting house price | Predicting spam or not spam |
| Function Used | Straight line equation | Sigmoid (S-shaped) function |
| Graph Shape | Straight line | S-curve |
| Error Measurement | Mean Squared Error (MSE) | Log Loss / Cross-Entropy |

### Term: Mean Squared Error (MSE)
A way to measure how wrong our predictions are, by squaring the difference between actual and predicted values, then averaging them. Used mostly in regression problems.

### Term: Log Loss (Cross-Entropy Loss)
A way to measure error in classification problems, especially when predicting probabilities. It punishes the model heavily when it is confidently wrong.

---

## 6. What is Model Evaluation?

**Definition (Simple):** Model Evaluation means checking **how good or bad** our trained model is at making correct predictions.

We can't just trust a model blindly — we need numbers/metrics to prove it's working well.

### Important Evaluation Terms:

- **Confusion Matrix:** A table that shows how many predictions were correct and incorrect, broken into 4 parts:
  - **True Positive (TP):** Predicted Yes, Actual Yes ✅
  - **True Negative (TN):** Predicted No, Actual No ✅
  - **False Positive (FP):** Predicted Yes, Actual No ❌ (Type 1 Error)
  - **False Negative (FN):** Predicted No, Actual Yes ❌ (Type 2 Error)

- **Accuracy:** How many predictions were correct out of all predictions.
    (the percentage of correct prediction)
  - Formula :  *Accuracy = (TP + TN) / (TP + TN + FP + FN)*

- **Precision:** Out of all things we predicted as "Yes", how many were actually "Yes"?
  - Useful when False Positives are costly (e.g., spam detection).
  - Formula : *Precision = TP / ( TP + FP )*

- **Recall (Sensitivity):** Out of all actual "Yes" cases, how many did we correctly catch/predicted?
  - Useful when False Negatives are dangerous (e.g., cancer detection — missing a real patient is very bad).
  - Formula :  *Recall= TP / ( FN + TP​ )*

- **F1 Score:** A balance between Precision and Recall (their harmonic mean). Useful when data is imbalanced.
  - Useful when you want balance between precision and recall.
  - Formula :  *F1 = ( 2 X Precision X Recall ) / ( Precision + Recall )*

**Simple takeaway:** Accuracy alone isn't always enough, especially with imbalanced data. That's why we also check Precision, Recall, and F1 Score.

---

## 7. K-Nearest Neighbors (KNN)

**Definition (Simple):** KNN is a classification algorithm that predicts the class of a new point by looking at the **"K" closest points (neighbors)** around it, and picks the class that appears most among them.
<br>

**The idea :** To guess something about a new data point, look at the points closest to it and copy the majority. <br>

**Real-life analogy :** <br>
You move into a new neighborhood and want to know if it's a "rich" or "middle-class" area. You don't check everyone — you just check your 5 nearest neighbors. If 4 out of 5 are rich, you guess "rich neighborhood."
<br>
That's it. That's KNN.


### How it works:
1. Choose a number "K" (e.g., K = 5).
2. When a new data point comes in, calculate its **distance** from all existing data points.
3. Pick the K closest (nearest) points.
4. See which class is the majority among those K points.
5. Assign that class to the new point.

**Example:** If you want to know if a fruit is an apple or orange, you check the 5 closest fruits (by size, color, weight) already labeled, and go with whichever label is more common among them.

### Term: Euclidean Distance
The straight-line distance between two points, calculated using a formula similar to Pythagoras' theorem. This is the most common way KNN measures "closeness."

### Term: Choosing K
- Small K (like K=1) → model can be sensitive to noise (overfitting).
- Large K → model becomes too general and may miss patterns (underfitting).
- We usually test different K values and pick the best one.

### Term: Overfitting
When a model learns the training data **too well**, including noise/errors, so it performs badly on new/unseen data.

### Term: Underfitting
When a model is **too simple** and fails to learn the actual patterns, performing badly on both training and testing data.

**Note:** KNN is called a "Lazy Learner" because it doesn't really "learn" during training — it just stores the data and does all the work at prediction time.

---

## 8. Naive Bayes

**Definition (Simple):** Naive Bayes is a classification algorithm based on **probability**. It uses **Bayes' Theorem** to predict the class of new data.

It's called "Naive" because it **assumes all features are independent** of each other (meaning one feature doesn't affect another), which is not always true in real life — but the algorithm still works surprisingly well.

### Term: Bayes' Theorem
A formula that calculates the probability of an event, based on prior knowledge of related events. In simple words: 

> "Given what I already know, what's the probability this new thing belongs to a certain class?"

### How it works (simple example):
Imagine classifying emails as Spam or Not Spam based on words. Naive Bayes calculates:
- The probability of a word appearing in Spam emails.
- The probability of a word appearing in Not Spam emails.
- Then it combines these probabilities for all words in the email to decide the final class.

### Term: Prior Probability
The probability of a class **before** looking at any features. Example: If 40% of all emails are spam, prior probability of spam = 0.4.

### Common Uses:
- Spam detection
- Sentiment analysis (positive/negative reviews)
- Document classification

---

## 9. Decision Tree

**Definition (Simple):** A Decision Tree is a classification algorithm that works like a **flowchart of questions**. It splits data step-by-step based on feature values until it reaches a final decision (class).

### How it works:
1. Start with the full dataset at the **Root Node**.
2. Ask a question about a feature (e.g., "Is age > 30?").
3. Split the data into branches based on the answer.
4. Keep splitting (asking more questions) until you reach a final answer — called a **Leaf Node**.

### Important Terms:
- **Root Node:** The very first question/split at the top of the tree.
- **Branch:** A path connecting one decision to the next.
- **Leaf Node:** The final output/class — no more splitting happens here.
- **Splitting:** The process of dividing data based on a condition.

### Term: Entropy
A measure of **randomness/impurity** in data. If a group has a mix of classes (like 50% yes, 50% no), entropy is high. If all data belongs to one class, entropy is 0 (pure).

### Term: Gini Index
Another way (like entropy) to measure impurity in data, often used because it's faster to calculate. Decision Trees use Entropy or Gini Index to decide the **best feature to split on** at each step.

### Term: Information Gain
How much "clarity" or "purity" we gain after splitting the data on a particular feature. The tree always picks the split with the **highest information gain**.

**Downside:** Decision Trees can easily **overfit** if they grow too deep (too many splits), so we often limit their depth or "prune" them (cut unnecessary branches).

---

## 10. Support Vector Machine (SVM)

**Definition (Simple):** SVM is a classification algorithm that tries to find the **best line (or boundary)** that separates different classes with the **maximum possible gap (margin)** between them.

### How it works:
1. Imagine plotting your data points on a graph.
2. SVM tries to draw a line (called a **hyperplane**) that separates the two classes.
3. Instead of just any line, SVM picks the line that has the **maximum distance** from the nearest points of both classes.

### Important Terms:

- **Hyperplane:** The decision boundary/line that separates classes. In 2D it's a line, in 3D it's a flat plane, and in higher dimensions it's called a hyperplane.

- **Margin:** The distance between the hyperplane and the nearest data points from each class. SVM tries to **maximize** this margin — a bigger margin usually means better generalization.

- **Support Vectors:** The data points that are **closest** to the hyperplane. These points are the most important because they "support" (define) where the boundary line goes. If you remove other points, the boundary won't change, but if you remove a support vector, it will.

- **Kernel Trick:** Sometimes data can't be separated by a straight line (it's mixed together in a curve pattern). The Kernel Trick is a technique that transforms the data into a higher dimension where it **becomes possible** to separate it with a straight line/hyperplane.
  - Common kernels: Linear, Polynomial, RBF (Radial Basis Function).

**Simple analogy:** Imagine separating red and blue marbles on a table using a stick. SVM doesn't just place the stick anywhere — it finds the position where the stick is **as far as possible** from the nearest marbles on both sides.

---

## 11. Applications of SVM

SVM is used in many real-world areas because it works well even with complex and high-dimensional data:

1. **Face Detection** — Identifying whether an image contains a face or not.
2. **Text and Document Classification** — Categorizing news articles, emails (spam detection), or documents into topics.
3. **Handwriting Recognition** — Used in recognizing handwritten digits/letters (like postal code reading).
4. **Bioinformatics** — Classifying genes, proteins, or detecting diseases like cancer from medical data.
5. **Image Classification** — Identifying objects within images (like distinguishing cats from dogs).
6. **Sentiment Analysis** — Determining if a review or comment is positive or negative.
7. **Stock Market Prediction** — Used in some financial models to classify market trends (though less common than regression models here).

---

## Quick Summary Table (All Algorithms)

| Algorithm | Best For | Key Idea |
|------------|-----------|----------|
| Logistic Regression | Simple binary classification | Uses Sigmoid function to output probability |
| KNN | Small datasets, simple patterns | Classifies based on closest neighbors |
| Naive Bayes | Text/spam classification | Uses probability (Bayes' Theorem) |
| Decision Tree | Easy-to-interpret decisions | Splits data using yes/no questions |
| SVM | Complex, high-dimensional data | Finds best boundary with max margin |

---

*End of Notes*