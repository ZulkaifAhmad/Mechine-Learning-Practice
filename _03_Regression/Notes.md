# Linear Regression — Complete Beginner Notes

---

## 1. What is Supervised Learning?

**Definition:**
Supervised Learning is a type of Machine Learning where the model learns from **labeled data** — meaning every training example has both an input (X) and a known correct output (y). The model learns the mapping `X → y` and then uses it to predict outputs for new, unseen inputs.

> Think of it like learning with an answer key. You show the model tons of examples where you already know the right answer, and it learns the pattern so it can answer correctly on new questions.

**Example:**
- Input: house size → Output: house price (you already have historical price data)
- Input: email text → Output: spam or not spam (you already have labeled emails)
- Input: hours studied → Output: exam score (you already have past student data)

**Analogy (JS mindset):**
Like training a function using a bunch of `input → expectedOutput` test cases, then letting it generalize to handle new inputs it hasn't seen before.

**How it works (high level):**
```
1. Collect labeled data (X, y)
2. Split into training data and test data
3. Train the model on training data (it learns the X → y pattern)
4. Test the model on unseen test data (check accuracy)
5. Use the trained model to predict y for brand-new X
```

### Types of Supervised Learning

Supervised learning splits into **two main types**, based on what kind of output (`y`) you're predicting:

| Type | Predicts | Output Type | Examples |
|------|----------|--------------|----------|
| **Regression** | A continuous numeric value | Number (any value in a range) | House price, salary, temperature, exam score |
| **Classification** | A category/class | Discrete label | Spam/Not spam, Cat/Dog, Pass/Fail, Disease/No disease |

**1. Regression**
Used when the answer is a **number** that can take any value on a continuous scale.
> Example: "How much will this house sell for?" → $245,000

**2. Classification**
Used when the answer is a **category** out of a fixed set of options.
> Example: "Is this email spam?" → Yes or No

**Sub-types of Classification (quick mention):**
| Sub-type | Meaning | Example |
|----------|---------|---------|
| **Binary Classification** | Only 2 possible classes | Spam vs Not spam |
| **Multiclass Classification** | More than 2 possible classes | Classifying an image as cat / dog / horse |

**Key distinction to remember:**
> If the output is a **number** → Regression.
> If the output is a **category/label** → Classification.

This is why Linear Regression (which you're learning now) falls under **Regression**, one of the two pillars of Supervised Learning. Now let's dive into it in detail.

---

## 2. What is Regression?

**Definition:**
Regression is a supervised machine learning technique used to predict a **continuous numeric value** based on input features.

> Unlike classification (which predicts categories like "spam/not spam"), regression predicts numbers like price, salary, temperature, marks, etc.

**Example:**
- Predicting house price based on size (sq ft)
- Predicting salary based on years of experience
- Predicting exam score based on hours studied

**Analogy (JS mindset):**
Think of it like a function `predictPrice(size)` that you're trying to *learn* from data, instead of hardcoding the logic yourself.

---

## 3. What is Linear Regression?

**Definition:**
Linear Regression assumes there is a **linear (straight-line) relationship** between the input (X) and output (y). <br>
Linear Regression is just the math that finds the one straight line that best passes through those dots.

**The Formula (Simple Linear Regression — 1 feature):**

```
y = mx + b
```

In ML notation, this is usually written as:

```
ŷ = θ₀ + θ₁x
```

| Symbol | Meaning |
|--------|---------|
| `ŷ` (y-hat) | Predicted value |
| `θ₀` (theta 0) | Intercept (bias) — value of y when x = 0 |
| `θ₁` (theta 1) | Slope — how much y changes when x increases by 1 |
| `x` | Input feature |

**Example:**
If you're predicting salary from years of experience:

```
salary = 30000 + 5000 × (years of experience)
```

- `θ₀ = 30000` → base salary with 0 experience
- `θ₁ = 5000` → salary increases by 5000 per year of experience

**Goal of Linear Regression:**
Find the best values of `θ₀` and `θ₁` so the predicted line fits the actual data points as closely as possible.

---

## 4. Multiple Linear Regression (Extension)

When you have more than one input feature:

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ... + θₙxₙ
```

**Example:**
Predicting house price using size, number of rooms, and location score:

```
price = θ₀ + θ₁(size) + θ₂(rooms) + θ₃(locationScore)
```

This is where the concept of a **hyperplane** comes in (explained below).

---

## 5. Cost Function

**Definition:**
The cost function measures **how wrong** the model's predictions are compared to actual values. It's a way to quantify the model's error.

**Why do we need it?**
We need a number that tells us "how bad" our current `θ₀` and `θ₁` are, so we can improve them.

**Most Common Cost Function: Mean Squared Error (MSE)**

```
J(θ₀, θ₁) = (1/2m) × Σ (ŷᵢ - yᵢ)²
```

| Symbol | Meaning |
|--------|---------|
| `J(θ)` | Cost function value |
| `m` | Number of training examples |
| `ŷᵢ` | Predicted value for example i |
| `yᵢ` | Actual value for example i |
| `Σ` | Sum over all training examples |

**Why squared?**
1. Removes negative signs (errors below and above the line both count as "error")
2. Penalizes larger errors more heavily than small ones

**Why divide by 2m?**
- Dividing by `m` gives the *average* error (so it's not affected by dataset size)
- Dividing by an extra `2` is just a mathematical convenience — it cancels out nicely when we take the derivative during gradient descent (explained next)

**Example:**
If your model predicts `$52,000` salary but actual is `$50,000`:
```
error = 52000 - 50000 = 2000
squared error = 2000² = 4,000,000
```

**Goal:** Minimize `J(θ)` — the smaller the cost, the better the model fits the data.

---

## 6. Gradient Descent

**Definition:**
Gradient Descent is an **optimization algorithm** used to find the values of `θ₀` and `θ₁` that minimize the cost function `J(θ)`.

**Intuition (Analogy):**
Imagine you're standing on a hill blindfolded and want to reach the lowest point (valley). You feel the slope under your feet and take a step in the direction that goes downhill. You repeat this until you can't go any lower — that's the minimum.

- The "hill" = the cost function graph
- "Taking a step downhill" = updating θ₀ and θ₁
- The "valley" = minimum cost (best-fit line)

**The Formula:**

```
θⱼ := θⱼ - α × (∂/∂θⱼ) J(θ)
```

| Symbol | Meaning |
|--------|---------|
| `:=` | "gets updated to" (assignment, not equality) |
| `α` (alpha) | Learning rate — how big each step is |
| `∂/∂θⱼ J(θ)` | Gradient (slope) of cost function w.r.t. θⱼ |

**For Linear Regression specifically:**

```
θ₀ := θ₀ - α × (1/m) × Σ (ŷᵢ - yᵢ)
θ₁ := θ₁ - α × (1/m) × Σ (ŷᵢ - yᵢ) × xᵢ
```

**What is the Learning Rate (α)?**
Controls how big each step is toward the minimum.

| Learning Rate | Effect |
|----------------|--------|
| Too small | Takes forever to converge (very slow training) |
| Too large | Might overshoot the minimum and never converge (bounces around or diverges) |
| Just right | Converges smoothly and efficiently |

**Important Rule:** Both `θ₀` and `θ₁` must be updated **simultaneously** (using old values), not one after another.

```python
# Correct (simultaneous update)
temp0 = theta0 - alpha * derivative0
temp1 = theta1 - alpha * derivative1
theta0 = temp0
theta1 = temp1
```

---

## 7. "Repeat Until Convergence" Theorem

**Definition:**
This refers to the iterative loop in gradient descent that keeps updating `θ₀` and `θ₁` repeatedly until the cost function stops decreasing significantly (i.e., it "converges" to the minimum).

**Pseudocode:**

```
repeat until convergence {
    θ₀ := θ₀ - α × (1/m) × Σ (ŷᵢ - yᵢ)
    θ₁ := θ₁ - α × (1/m) × Σ (ŷᵢ - yᵢ) × xᵢ
}
```

**What does "convergence" mean?**
Convergence happens when:
- The cost function `J(θ)` stops decreasing significantly between iterations, OR
- You reach a fixed maximum number of iterations (a safety limit), OR
- The change in θ values becomes smaller than a very tiny threshold (e.g. `0.0001`)

**Example (JS-style loop analogy):**

```javascript
let theta0 = 0, theta1 = 0;
const alpha = 0.01;

for (let i = 0; i < 1000; i++) {
  // compute gradients
  // update theta0, theta1
  // if change is very small, break early (convergence reached)
}
```

**Why "repeat"?**
A single update step won't reach the minimum — it takes many small steps, gradually reducing the cost each time, like walking downhill step by step rather than jumping straight to the bottom.

---

## 8. Hyperplane

**Definition:**
A hyperplane is the **generalized version of a line** in higher-dimensional space. It's the "decision surface" or "prediction surface" that the regression model fits to the data.

| Number of Features | What the model draws |
|---------------------|------------------------|
| 1 feature | A straight **line** (2D) |
| 2 features | A flat **plane** (3D) |
| 3+ features | A **hyperplane** (cannot be visualized, but mathematically the same idea) |

**Formula for a hyperplane (general form of multiple linear regression):**

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

**Example (2 features → 3D plane):**
Predicting price using size and number of rooms — the model doesn't fit a line anymore, it fits a **flat plane** floating in 3D space, where every point on the plane is a predicted price for a given (size, rooms) pair.

**Example (many features → hyperplane):**
Predicting a house price using 10 features (size, rooms, location, age, etc.) means the model is fitting a hyperplane in **10-dimensional space** — you can't draw it, but the math works exactly the same way as a 2D line.

**Key takeaway:**
> "Line" and "Hyperplane" are the same concept — just at different dimensions. Linear regression always tries to fit the best hyperplane through the data.

---

## 9. Putting It All Together — Full Workflow

```
1. Start with random θ₀, θ₁ (often initialized to 0)
2. Use the model: ŷ = θ₀ + θ₁x   → make predictions
3. Calculate the Cost Function J(θ) → measure how wrong predictions are
4. Use Gradient Descent → adjust θ₀, θ₁ to reduce the cost
5. Repeat steps 2-4 until convergence (cost stops improving)
6. Final θ₀, θ₁ define your trained model's line/hyperplane
```

---

## 10. Small but Important Related Terms

### Slope
**Definition:** The slope (`θ₁` / `m`) tells you how steep the line is — how much `y` changes for every 1-unit increase in `x`.

```
slope = (change in y) / (change in x) = Δy / Δx
```

**Example:** If slope = 5000, every extra year of experience adds $5000 to predicted salary. A bigger slope = steeper line = faster change.

---

### Intercept
**Definition:** The value of `y` when `x = 0` — where the line crosses the y-axis. Also called the **bias** term.

**Example:** If `θ₀ = 30000`, someone with 0 years of experience is predicted to earn $30,000.

---

### Best Fit Line
**Definition:** The one specific line (out of infinite possible lines) that minimizes the cost function — i.e., the line that sits as close as possible to all the data points overall.

> This is literally the *goal* of linear regression: finding the best fit line by tuning θ₀ and θ₁.

**Analogy:** Imagine scattering dots on a graph, then trying to draw one straight line through them so the total distance from all dots to the line is as small as possible. That line = best fit line.

---

### Residual (Residual Error)
**Definition:** The difference between the actual value and the predicted value for a single data point. Also just called **"error"**.

```
residual = yᵢ - ŷᵢ
```

**Example:** Actual salary = $52,000, predicted = $50,000 → residual = $2,000.

- Positive residual → model **underpredicted** (actual is higher than prediction)
- Negative residual → model **overpredicted** (actual is lower than prediction)
- Residual = 0 → perfect prediction for that point

**Note:** Residuals are what you square and average to get the Cost Function (MSE). So residual = error for one point, cost = combined error for all points.

---

### Residual Plot
**Definition:** A scatter plot of residuals (y-axis) vs. predicted values or x (x-axis). Used to check if linear regression is a good fit.

- Residuals scattered **randomly around 0** → linear model is a good fit
- Residuals show a **pattern/curve** → data isn't actually linear, a different model may fit better

---

### Outlier
**Definition:** A data point that lies far away from the general trend of the rest of the data. Outliers have large residuals and can heavily distort the best fit line since MSE punishes big errors a lot (because of squaring).

**Example:** In salary data, one person earning $500,000 with only 1 year of experience would badly pull the line toward itself.

---

### Overfitting vs Underfitting (quick preview)
| Term | Meaning |
|------|---------|
| **Underfitting** | Line is too simple, doesn't capture the trend well (high error on training data itself) |
| **Overfitting** | Model fits training data *too* perfectly, including noise — performs poorly on new/unseen data |

*(These matter more once you move to multiple features / polynomial regression — just good to know the terms early.)*

---

### R² Score (Coefficient of Determination) — sneak peek
**Definition:** A metric (0 to 1) that tells you how well your regression line explains the variation in the data. Closer to 1 = better fit.

```
R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)
```

**Example:** `R² = 0.85` means your model explains 85% of the variation in the output — decent fit. (We'll cover this in depth separately.)

---

## 11. Quick Reference Cheatsheet

| Term | One-Line Definition |
|------|----------------------|
| **Regression** | Predicting continuous numeric values |
| **Linear Regression** | Fitting a straight-line relationship between input and output |
| **θ₀ (Intercept)** | Base value when x = 0 |
| **θ₁ (Slope)** | Rate of change of y with respect to x |
| **Cost Function J(θ)** | Measures how wrong the model's predictions are (avg squared error) |
| **Gradient Descent** | Algorithm to minimize cost by iteratively adjusting θ values |
| **Learning Rate (α)** | Step size in gradient descent |
| **Convergence** | Point where cost stops meaningfully decreasing |
| **Hyperplane** | Generalized "line" in higher dimensions for multiple features |

---

## 12. Common Beginner Confusions (Clarified)

**Q: Is cost function the same as loss function?**
Loss = error for ONE training example. Cost = average loss across ALL training examples. They're related but not identical terms.

**Q: Why not just solve for θ directly using math (Normal Equation) instead of Gradient Descent?**
You can, for small datasets (Normal Equation is a closed-form solution). But Gradient Descent scales much better for large datasets with many features, which is why it's the standard approach in ML/DL.

**Q: Does gradient descent always find the best solution?**
For linear regression, yes — the cost function is convex (bowl-shaped), so gradient descent will always converge to the **global minimum** if the learning rate is set well.