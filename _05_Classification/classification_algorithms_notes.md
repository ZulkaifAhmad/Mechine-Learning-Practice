# Classification Algorithms — Flow Notes

> Simple definitions, visual graphs, formulas, and what everything means in plain English. No heavy math — just how each model works.

---

## Table of Contents
1. [Logistic Regression](#1-logistic-regression)
2. [Model Evaluation (Confusion Matrix & Metrics)](#2-model-evaluation-confusion-matrix--metrics)
3. [KNN (K-Nearest Neighbour)](#3-knn-k-nearest-neighbour)
4. [Naive Bayes](#4-naive-bayes)
5. [Decision Tree](#5-decision-tree)
6. [SVM (Support Vector Machine)](#6-svm-support-vector-machine)

---

## 1. Logistic Regression

### Simple Definition
A math formula that predicts if something is a "Yes" or a "No". 

### What it means in plain English
It takes a bunch of data and squashes it into a smooth "S" shape. If a new point lands on the top half of the "S", the model says "Yes" (1). If it lands on the bottom half, the model says "No" (0).

### The Graph
```text
  1 |       ------- (Yes / 100% chance)
    |      /
  P |     /
    |    /  <-- Threshold (Usually 50%)
  0 |----           (No / 0% chance)
    ------------------
```

### The Beginner's "Why"
**Why do we use the Sigmoid function?** 
Imagine drawing a straight line to predict a Yes/No answer. That straight line goes up to infinity and down to negative infinity. But probabilities *must* live between 0% and 100% (0 and 1). The sigmoid function is basically a mathematical trash compactor—it takes that infinite straight line and squashes it down into a smooth "S" shape so that no matter how big or small the number is, the output is always safely trapped between 0 and 1.

### Key Terminology
- **Sigmoid Function** — the math rule that makes the S-shape.
- **Euler's number (e)** — a constant (around 2.718), used to create the smooth curve.
- **Hypothesis** $h_\theta(x)$ — the model's final guess (the probability).
- **Threshold** — the cutoff point (like 0.5) to decide if it's a Yes or No.
- **Log Loss** — the score of how badly the model messed up.
- **Global Minima** — the point where mistakes are as low as possible.

### Formulas

**1) Straight line:**
$$z = \theta_0 + \theta_1 x_1$$

| Symbol | Meaning |
|---|---|
| $\theta_0$ | intercept / bias — where the line starts |
| $\theta_1$ | weight/coefficient — how much $x_1$ affects the result |
| $x_1$ | input feature (e.g., weight of a person) |

**2) Sigmoid function (squashes the line):**
$$g(z) = \frac{1}{1 + e^{-z}}$$

**3) Final guess (hypothesis):**
$$h_\theta(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1)}}$$

**4) Log Loss (measuring the mistakes):**
$$\text{Log Loss} = -\frac{1}{m} \sum [ y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)}) ]$$

### Flow of the Algorithm
1. Take the data and compute a raw score.
2. Pass that score through the sigmoid function to get a probability between 0 and 1.
3. Compare the probability to 0.5 to assign a Yes or No.
4. Check how wrong the guess was using Log Loss.
5. Adjust the math until the mistakes are as small as possible.

---

## 2. Model Evaluation (Confusion Matrix & Metrics)

### Simple Definition
A scorecard that tells you exactly how your model got things right and how it got things wrong.

### What it means in plain English
Instead of just saying "The model is 90% right," the confusion matrix breaks it down. It tells you "It guessed Yes correctly 80 times, but it accidentally guessed Yes 10 times when it shouldn't have."

### The Graph
```text
              Predicted YES     Predicted NO
            +-----------------+-----------------+
 Actual YES | True Positive   | False Negative  |
            | (Hit!)          | (Missed it)     |
            +-----------------+-----------------+
 Actual NO  | False Positive  | True Negative   |
            | (False Alarm)   | (Correctly NO)  |
            +-----------------+-----------------+
```

### The Beginner's "Why"
**Why do we need Precision and Recall instead of just Accuracy?** 
Accuracy is bad if things are unbalanced. If 99% of emails are good and 1% is spam, a model that guesses "good" every time is 99% accurate—but completely useless for stopping spam. 
*   **Precision:** Use this when false alarms are terrible (you don't want a vital work email marked as spam). 
*   **Recall:** Use this when missing a target is terrible (you don't want a doctor to miss a cancer diagnosis, even if it causes a few false alarms).

### Formulas

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall:**
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score:** (A balance between Precision and Recall)
$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 3. KNN (K-Nearest Neighbour)

### Simple Definition
A guessing method that looks at the closest neighbors to make a decision.

### What it means in plain English
"Birds of a feather flock together." If you find a new dot on a map and you want to know what color it is, you look at the 3 dots closest to it. If 2 of them are Red and 1 is Blue, the new dot is probably Red.

### The Graph
```text
   (Red)       (Red)
         \    /
          ( ? ) ---> It's closest to Red, so it's Red!
         /     
   (Red)               (Blue)
```

### Key Terminology
- **K** — how many neighbors you want to look at (pick an **odd number** like 3 or 5 so there are no ties).
- **Euclidean Distance** — drawing a straight line with a ruler to see how far apart two dots are.

### Formula

**Euclidean Distance:**
$$\text{distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### Flow of the Algorithm
1. Pick a number for K (like 3).
2. Measure the distance from the new dot to every other dot.
3. Find the 3 closest dots.
4. Take a vote. Whichever color shows up the most wins.

---

## 4. Naive Bayes

### Simple Definition
A guessing method based on past clues and percentages.

### What it means in plain English
It acts like a detective looking at past evidence. If you see an animal that barks, has fur, and chases cats, Naive Bayes looks at past data and says: "95% of the time I saw these clues in the past, it was a dog." It assumes every clue is separate (naive) to keep the math easy.

### The Graph
```text
   Clue 1 (Bark) ----\
   Clue 2 (Fur) ------+---> 95% Chance it's a Dog!
   Clue 3 (Tail) ----/
```

### Key Terminology
- **Conditional Probability** — the chance of something happening, given a clue. (E.g., what is the chance it's a dog, *given* that it barks?)
- **Bayes' Theorem** — the math rule used to calculate these clues.

### Formulas

**Bayes' Theorem:**
$$P(A|B) = \frac{P(A) \times P(B|A)}{P(B)}$$

| Symbol | Meaning |
|---|---|
| $P(A\|B)$ | probability of A happening, given B already happened |
| $P(A)$ | probability of A happening on its own |

**Applied to machine learning:**
$$P(y | x_1,x_2,x_3) = \frac{P(y) \times P(x_1|y) \times P(x_2|y) \times P(x_3|y)}{P(x_1) \times P(x_2) \times P(x_3)}$$

### Flow of the Algorithm
1. Look at past data and find the odds of each class (e.g., how many dogs vs cats exist total).
2. Look at each clue separately and find its odds (how many dogs bark?).
3. Multiply the odds together for "Dog" and then for "Cat".
4. Whichever score is higher wins.

---

## 5. Decision Tree

### Simple Definition
A flowchart of Yes/No questions that leads to an answer.

### What it means in plain English
It plays the game "20 Questions." It looks at the data and figures out the best possible question to ask first to split the group up. It keeps asking questions until it is completely sure of the answer.

### The Graph
```text
         [Is it raining outside?]  <-- Root (Best first question)
           /                 \
        (Yes)                (No)
         /                     \
   [Take Umbrella]         [Wear Sunglasses]  <-- Leaves (Final answers)
```

### The Beginner's "Why"
**Why use Entropy and Information Gain?**
You wouldn't start a guessing game by asking, "Is the person wearing a red shirt with three buttons?" That's too specific. You start with "Are they male or female?" to cut the crowd in half. 
*   **Entropy** measures how mixed up a group is. 
*   **Information Gain** tells the computer which question cleans up that mess the fastest.

### Formulas

**Entropy (Messiness):**
$$\text{Entropy}(S) = -P(+) \log_2(P(+)) - P(-) \log_2(P(-))$$

**Information Gain (Clearing the mess):**
$$IG(S, \text{Feature}) = \text{Entropy}(S) - \sum \left( \frac{|S_v|}{|S|} \times \text{Entropy}(S_v) \right)$$

### Flow of the Algorithm
1. Calculate the Information Gain for every possible question.
2. Pick the question with the highest score (the one that splits the data best).
3. Draw branches for "Yes" and "No".
4. Keep picking the next best question until you have a final answer.

---

## 6. SVM (Support Vector Machine)

### Simple Definition
A straight line that acts like a wide fence between two groups.

### What it means in plain English
Imagine a two-lane road with oncoming traffic. You don't want to drive exactly on the center line, and you don't want to drive in the ditch. You want to stay exactly in the middle of your lane to give yourself the biggest safety buffer. SVM draws a line perfectly in the middle of two groups so the gap (safety buffer) is as wide as possible.

### The Graph
```text
   Group 1 (X)         Group 2 (O)
  
   X    X     |   |     O
              |   |
      X       | / |        O
              |/  |    O
   X          |   |  
             Gap (Margin)
```

### Key Terminology
- **Hyperplane** — the main dividing line.
- **Margin** — the empty gap (the safety buffer) around the line.
- **Support Vectors** — the dots closest to the line. They are the only ones that matter for drawing the line.
- **Kernel** — a math trick. If you can't draw a straight line between the dots, the Kernel bends the map into 3D so you can drop a flat sheet between them.

### Formula

**Equation of the line:**
$$w \cdot x + b = 0$$

**Decision rule:**
- If $w \cdot x + b > 0 \rightarrow$ Class 1
- If $w \cdot x + b < 0 \rightarrow$ Class 2

### Flow of the Algorithm
1. Plot all the dots.
2. Draw a line that separates the two groups.
3. Draw "margin" lines touching the closest dots on each side.
4. Shift the line until the gap between the margins is perfectly maximized.
5. If a straight line doesn't work, use a Kernel to bend the data into 3D.
