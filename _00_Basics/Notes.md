# Machine Learning Basics 

Simple notes to understand AI, ML, DL, and Data Science — written so you can explain them confidently in an interview.

---

## 1. Artificial Intelligence (AI)

**Definition:** AI is the broad idea of making machines smart enough to do tasks that normally need human intelligence — like thinking, deciding, and solving problems.

**Explanation:** Think of AI as the biggest umbrella. Everything else (ML, DL) sits inside it.

**Example:** Siri answering your questions, a chess-playing computer, Netflix suggesting movies.

---

## 2. Machine Learning (ML)

**Definition:** ML is a part (subset) of AI where we teach a computer to learn from data and improve automatically, without writing exact rules for every situation.

**Explanation:** Instead of telling the computer "if this happens, do that," we give it lots of examples (data), and it finds the pattern itself.

**Example:** Showing a model 1000 photos of cats and dogs. It learns the pattern and can now guess "cat" or "dog" for a new photo.

**Interview line:** "ML is a subset of AI that allows systems to learn patterns from data and make predictions without being explicitly programmed."

---

## 3. Deep Learning (DL)

**Definition:** DL is a part (subset) of ML that uses **neural networks** (inspired by the human brain) with many layers to learn from large amounts of data.

**Explanation:** DL is more powerful than normal ML for complex tasks like images, speech, and text, but it needs a lot of data and computing power.

**Example:** Face recognition on your phone, self-driving cars detecting pedestrians, ChatGPT understanding language.

**Relationship (important for interview):**
```
AI  →  ML  →  DL
(Broadest)     (Most specific)
```

---

## 4. Data Science

**Definition:** Data Science is the field of collecting, cleaning, analyzing, and visualizing data to find useful insights and help in decision-making. ML is one of the tools Data Science uses.

**Explanation:** A Data Scientist's job isn't only building ML models — it also includes understanding data, finding trends, and telling a "story" from numbers.

**Example:** A company analyzing customer purchase data to decide which product to promote next.

**Simple difference to remember:**
| Field | Main Focus |
|---|---|
| AI | Making machines intelligent |
| ML | Learning from data |
| DL | Learning using neural networks |
| Data Science | Extracting insights from data (uses ML as a tool) |

---

## 5. Types of Machine Learning

ML has **3 main types**, based on how the model learns:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

---

### 5.1 Supervised Learning

**Definition:** The model learns from **labeled data** — meaning every input already has the correct output (answer) given to it.

**Explanation:** It's like a teacher supervising a student — you show the question AND the answer, so the model learns the relationship between them.

*Label Data : Data where every input already has the correct output (answer) attached to it.*

*Example: A photo of an animal tagged with "Cat" or "Dog" — the answer is already given, so the model learns by matching input to that known answer.*


**Two types:**
- **Classification** → predicts a category (Yes/No, Cat/Dog, Spam/Not Spam)
- **Regression** → predicts a number (price, temperature, salary)

**Examples:**
- Email spam detection (Spam or Not Spam) → Classification
- Predicting house price based on size and location → Regression
- Predicting if a student will pass or fail based on study hours → Classification

**Interview line:** "In supervised learning, we train the model on labeled data, where input and output pairs are known, so the model learns to map input to output."

---

### 5.2 Unsupervised Learning

**Definition:** The model learns from **unlabeled data** — there is no correct answer given. The model finds hidden patterns or groups on its own.

**Explanation:** we give the machine unlabeled data (no answers/tags), and the machine looks at the features/similarities in that input data, then groups similar items together on its own — without us telling it what the groups should be.

*Unlabeled Data : Data where every input already has the correct output (answer) attached to it.*

*Example: A photo of an animal tagged with "Cat" or "Dog" — the answer is already given, so the model learns by matching input to that known answer.*



**Two common types:**
- **Clustering** → grouping similar data (e.g., grouping customers)
- **Association** → finding relationships between items (e.g., people who buy bread also buy butter)

**Examples:**
- Grouping customers based on shopping behavior (Clustering) — e.g., K-Means
- Market basket analysis: "customers who buy X also buy Y"
- Grouping news articles by topic without knowing the topics in advance

**Interview line:** "In unsupervised learning, we work with unlabeled data and the model tries to find hidden patterns or groupings on its own, without predefined outputs."

---

### 5.3 Reinforcement Learning

**Definition:** The model (called an **agent**) learns by **trial and error**, taking actions in an environment and getting **rewards** (for good actions) or **penalties** (for bad actions).

**Explanation:** Reinforcement learning doesn't use a pre-collected dataset like supervised or unsupervised learning — instead, Reinforcement learning is when an agent learns by interacting with an environment — taking actions and receiving rewards for good actions or penalties (negative feedback or punishment) for bad ones — to gradually learn the best strategy to maximize reward.

**Key terms:**
- **Agent** → the learner/decision maker
- **Environment** → the world the agent interacts with
- **Reward** → feedback for a good/bad action

**Examples:**
- A robot learning to walk (falls = penalty, walks = reward)
- AI playing chess or video games and improving with each game
- Self-driving car learning to stay in lane (reward) vs hitting a curb (penalty)

**Interview line:** "Reinforcement learning is where an agent learns by interacting with an environment, receiving rewards for good actions and penalties for bad ones, to maximize long-term reward."

---

### Quick Comparison Table

| Type | Data Used | Goal | Example |
|---|---|---|---|
| Supervised | Labeled data | Predict output | Spam detection |
| Unsupervised | Unlabeled data | Find patterns/groups | Customer segmentation |
| Reinforcement | No fixed data, learns from rewards | Maximize reward | Game-playing AI |

---

## 6. EDA (Exploratory Data Analysis)

**Definition:** EDA is the process of exploring and understanding your dataset **before** building any ML model — checking its structure, patterns, missing values, and relationships.

**Explanation:** Before cooking, you check your ingredients — same way, before building a model, you check your data. It answers: Is data clean? Are there missing values? Are there outliers? What's the relationship between columns?

**Common steps in EDA:**
1. Check data shape (rows & columns) and data types
2. Check for missing values
3. Check for duplicates
4. Check for outliers (unusual values)
5. Visualize data (histograms, box plots, scatter plots, correlation heatmap)
6. Understand relationships between features

**Example:** Before predicting house prices, you'd check: Are there missing "price" values? Does "size" correlate with "price"? Are there any weird values like negative prices?

**Interview line:** "EDA is the process of analyzing datasets to summarize their main characteristics, often using visualizations, to understand patterns, spot anomalies, and check assumptions before modeling."

---

## 7. A Few Extra Important Terms (Bonus)

Since you're starting ML, these terms come up a lot — good to know early:

| Term | Simple Meaning |
|---|---|
| **Feature** | An input column/variable used to predict (e.g., size, age) |
| **Feature Scalling** | Feature scaling means adjusting the range of your features so they're all on a similar scale, instead of some being very large numbers and others very small. |
| **Label / Target** | The output/answer we want to predict (e.g., price) |
| **Model** | The trained "brain" that makes predictions |
| **Training Data** | Data used to teach the model |
| **Testing Data** | New data used to check how well model learned |
| **Overfitting** | Model memorizes training data too well, performs badly on new data |
| **Underfitting** | Model is too simple, doesn't learn patterns well even on training data |
| **Accuracy** | How many predictions were correct (used in classification) |

**Interview tip:** If asked "what's overfitting vs underfitting?" say:
- **Overfitting** = model does great on training data but poor on new/test data (memorized instead of learned)
- **Underfitting** = model does poorly on both training and test data (too simple to learn the pattern)



---

## Data Pre-Processing has 3 Main Parts

1. Data Cleaning
2. Data Transformation
3. Data Reduction / Splitting

---

## 1. Data Cleaning

**Definition:** Fixing the "dirty" or messy parts of the data — missing values, duplicates, and wrong/outlier values.

### 1.1 Handling Missing Values
**Definition:** Filling in or removing empty/missing data points.
**Example:** If "Age" is missing for some rows, fill it with the average age (or drop that row).

### 1.2 Removing Duplicates
**Definition:** Deleting repeated/duplicate rows from the dataset.
**Example:** The same customer's record accidentally appearing twice in the data.

### 1.3 Handling Outliers
**Definition:** Fixing or removing extreme/unusual values that don't make sense.
**Example:** An "Age" column showing 500 years — clearly wrong, needs fixing or removing.

### 1.4 Fixing Incorrect / Inconsistent Data
**Definition:** Correcting wrong formats or inconsistent entries.
**Example:** Some rows have "Male"/"Female" and others have "M"/"F" — need to make them consistent.

---

## 2. Data Transformation

**Definition:** Converting data into a format/structure that a model can better understand and work with.

### 2.1 Encoding Categorical Data
**Definition:** Converting text/category values into numbers, since ML models understand numbers, not words.
**Example:** "Male"/"Female" → converted to 0/1. "Red"/"Blue"/"Green" → converted using One-Hot Encoding.

### 2.2 Feature Scaling
**Definition:** Adjusting the range of features so they're all on a similar scale, so no feature dominates just because of bigger numbers.
**Example:** Age (18–60) and Salary (20,000–200,000) both scaled to a 0–1 range.
- **Normalization** → scales values between 0 and 1
- **Standardization** → centers values around 0 with a standard spread

### 2.3 Feature Engineering (bonus, often part of transformation)
**Definition:** Creating new useful features from existing data to help the model learn better.
**Example:** From a "Date of Birth" column, creating a new "Age" column.

---

## 3. Data Reduction / Splitting

### 3.1 Dimensionality Reduction
**Definition:** Reducing the number of features while keeping the important information — useful when data has too many columns.
**Example:** Compressing 100 columns into 10 key columns using PCA (Principal Component Analysis).

### 3.2 Train-Test Split
**Definition:** Dividing the dataset into a **Training Set** (to teach the model) and a **Testing Set** (to check how well it learned).
**Example:** 80% of data used for training, 20% kept aside for testing.

---

## Quick Summary Table

| Step | Purpose | Example |
|---|---|---|
| Handling Missing Values | Fill/remove empty data | Fill missing age with average |
| Removing Duplicates | Remove repeated rows | Delete duplicate customer entry |
| Handling Outliers | Fix extreme wrong values | Fix "Age = 500" |
| Encoding Categorical Data | Convert text to numbers | "Male"/"Female" → 0/1 |
| Feature Scaling | Bring features to same range | Age & Salary both 0–1 |
| Feature Engineering | Create new useful features | DOB → Age |
| Dimensionality Reduction | Reduce number of features | 100 columns → 10 (PCA) |
| Train-Test Split | Split data for training/testing | 80% train, 20% test |

---

## Data Cleaning vs Data Pre-Processing (common confusion)

| Term | What it means |
|---|---|
| **Data Cleaning** | Only fixing dirty parts — missing values, duplicates, outliers |
| **Data Pre-Processing** | The full journey — cleaning + transformation (encoding, scaling) + splitting |

**Analogy:** Data Cleaning = washing the vegetables. Data Pre-Processing = washing + cutting + measuring + arranging everything ready to cook (train the model).

**Interview line:** "Data cleaning is just one step within data pre-processing — cleaning fixes issues like missing values and duplicates, while pre-processing also includes transforming data and splitting it for training and testing."

---

## Summary (30-second interview answer)

> "Data pre-processing is the process of preparing raw data before training an ML model. It has three main parts — data cleaning (handling missing values, duplicates, and outliers), data transformation (encoding categorical data and feature scaling), and data reduction/splitting (dimensionality reduction and train-test split). This step is important because clean, well-prepared data directly improves model accuracy."

---

---

## Summary (30-second interview answer)

> "AI is the broad goal of making machines intelligent. ML is a subset of AI where machines learn patterns from data. DL is a subset of ML that uses neural networks for complex tasks. Data Science uses ML and statistics to extract insights from data. ML itself has three types — supervised (learns from labeled data), unsupervised (finds patterns in unlabeled data), and reinforcement learning (learns through rewards and penalties). Before building any ML model, we perform EDA to understand and clean the data."

---
*Keep this file in your ML basics folder and revise it before interviews.*