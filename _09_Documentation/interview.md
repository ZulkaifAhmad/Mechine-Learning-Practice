# Machine Learning — 50 Interview Questions

These 50 questions go from **basic to advanced**, covering everything in your notes (EDA, ML Basics, Regression, Classification, Model Tuning, Ensemble Learning, Unsupervised Learning).
Use this to test yourself. Try answering out loud or in writing before checking the highly detailed Answer Key at the bottom.

## Table of Contents

- [Questions](#questions)
  - [Section A: EDA & Preprocessing (Q1-9)](#section-a-eda--preprocessing)
  - [Section B: ML Basics (Q10-15)](#section-b-ml-basics)
  - [Section C: Regression (Q16-25)](#section-c-regression)
  - [Section D: Classification (Q26-36)](#section-d-classification)
  - [Section E: Model Tuning (Q37-40)](#section-e-model-tuning)
  - [Section F: Ensemble Learning (Q41-45)](#section-f-ensemble-learning)
  - [Section G: Unsupervised Learning (Q46-50)](#section-g-unsupervised-learning)
- [Answer Key](#answer-key)
  - [Answers: Section A: EDA & Preprocessing](#answers-section-a-eda--preprocessing)
  - [Answers: Section B: ML Basics](#answers-section-b-ml-basics)
  - [Answers: Section C: Regression](#answers-section-c-regression)
  - [Answers: Section D: Classification](#answers-section-d-classification)
  - [Answers: Section E: Model Tuning](#answers-section-e-model-tuning)
  - [Answers: Section F: Ensemble Learning](#answers-section-f-ensemble-learning)
  - [Answers: Section G: Unsupervised Learning](#answers-section-g-unsupervised-learning)

---

## Questions

### Section A: EDA & Preprocessing

1. What is EDA (Exploratory Data Analysis), and why do we do it before building a model?
2. What steps do you typically perform during EDA on a new dataset?
3. What is the difference between Data Cleaning and Feature Engineering (the process of creating new useful features from existing data)?
4. How do you handle missing values in a dataset? Name at least 3 different strategies.
5. When would you use median imputation instead of mean imputation for filling missing values?
6. What is an outlier, and how would you detect one using the IQR (Interquartile Range) method?
7. What is the difference between Normalization and Standardization? When would you use each?
8. What is the difference between Label Encoding and One-Hot Encoding, and when should you use each?
9. What is Binning, and can you give an example of when it would be useful?

### Section B: ML Basics

10. In your own words, what is Machine Learning?
11. What is the difference between Supervised Learning and Unsupervised Learning?
12. What is the difference between Regression and Classification?
13. What is Bias, and what is Variance, in the context of a model's errors?
14. What is Underfitting? What is Overfitting? How would you fix each one?
15. Explain the Bias-Variance Tradeoff in your own words.

### Section C: Regression

16. What is Linear Regression, and what equation does it use?
17. What is a Residual Error?
18. What is a Cost Function, and why is MSE (Mean Squared Error) commonly used for regression?
19. Explain how Gradient Descent works, step by step.
20. What happens if the Learning Rate is too high? What happens if it's too low?
21. What is Multi-Linear Regression, and what problem (Multicollinearity) should you watch out for?
22. What is a Hyperplane, and how does it relate to Multi-Linear Regression and SVM?
23. What is R², and why do we sometimes prefer Adjusted R² over plain R²?
24. What is Polynomial Regression, and how is it different from Linear Regression?
25. What is Regularization? Explain the difference between Ridge and Lasso Regression.

### Section D: Classification

26. What is Logistic Regression, and why is it used for classification even though it has "regression" in the name?
27. What does the Sigmoid Function do, and what is its output range?
28. What is a Decision Boundary?
29. What are the key differences between Linear Regression and Logistic Regression?
30. Why isn't MSE typically used as the loss function for Logistic Regression? What is used instead?
31. What is Entropy, and how is it used in building a Decision Tree?
32. Explain how the KNN algorithm makes a prediction, step by step.
33. What is Euclidean Distance, and why is scaling important for KNN?
34. What does "naive" mean in Naive Bayes, and why does the algorithm still work well despite this assumption?
35. Explain how a Decision Tree decides where to split at each node (Information Gain / Gini Index).
36. What is SVM, and what are "support vectors"? What is the purpose of the Kernel Trick?

### Section E: Model Tuning

37. What is the difference between a Parameter and a Hyperparameter?
38. What is Cross Validation, and why is it more reliable than a single Train-Test Split?
39. What is the difference between Grid Search CV and Random Search CV? When would you choose one over the other?
40. Why is Accuracy sometimes a misleading metric, and what other metrics (Precision, Recall, F1 Score, ROC-AUC) would you use instead? Give an example of when Precision matters more than Recall, and vice versa.

### Section F: Ensemble Learning

41. What is Ensemble Learning, and why does combining multiple models often work better than one model alone?
42. What is the difference between Bagging and Boosting?
43. Explain how Random Forest works, and why it usually performs better than a single Decision Tree.
44. Explain the difference between how AdaBoost and Gradient Boosting each improve on previous models' mistakes.
45. What makes XGBoost different from (and often better than) standard Gradient Boosting?

### Section G: Unsupervised Learning

46. What is Clustering, and how is it different from Classification?
47. Explain the K-Means algorithm step by step. How do you choose the right value of K (Elbow Method, Silhouette Score)?
48. What is the difference between K-Means, Hierarchical Clustering, and DBSCAN? When would you use each?
49. What is PCA, and how is it different from Feature Selection? What does it mean when we say principal components are "hard to interpret"?
50. What is the difference between Anomaly Detection and normal outlier removal during EDA? Can you give a real-world use case for Anomaly Detection?

---

<br><br>

# Answer Key

Welcome to the simple, beginner-friendly answer key! These explanations are written in very simple English, just like explaining to a 10-year-old, so you can easily understand what is going on.

### Answers: Section A: EDA & Preprocessing

**1. What is EDA, and why do we do it?**
EDA is like checking your toys before playing. We want to see if anything is broken or missing. If we try to play with broken toys, the game is ruined. We check the data so our computer doesn't get confused!

**2. What steps do you typically perform during EDA?**
First, we count how many things we have. Then, we look to see if any pieces are missing or doubled. We also draw simple pictures (like bar charts) to see what the data looks like overall.

**3. Data Cleaning vs. Feature Engineering?**
Data cleaning is fixing broken toys (like wiping off dirt or taping a broken piece). Feature engineering is building a cool new toy out of old ones, like snapping two Lego blocks together to make a bigger, better one!

**4. Handling missing values (3 strategies)?**
1. **Throw it away:** If a toy is missing a piece, just throw the whole toy out (delete the row).
2. **Put a normal piece there:** Fill the empty spot with the most normal, average piece you can find (mean/median).
3. **Guess:** Look at other toys that look similar and guess what piece should belong there.

**5. Median imputation vs. Mean imputation?**
If you have 9 kids with 1 toy and 1 kid with 100 toys, the "mean" (average) says everyone has 10 toys, which is a lie! The "median" lines everyone up and looks at the kid in the middle, who has 1 toy. The median is more honest when you have crazy big numbers.

**6. What is an outlier, and how does the IQR method find them?**
An outlier is a total weirdo, like a dog in a room full of cats. The IQR rule says we only look at the middle 50% of normal cats. If an animal is way too far away from these normal cats, it's an outlier!

**7. Normalization vs. Standardization?**
Both shrink giant numbers so they are easier to look at. 
- **Normalization:** Squishes everything to be exactly between 0 and 1, like shrinking everything to fit on a ruler.
- **Standardization:** Just moves the middle of the numbers to exactly 0, keeping their original shape.

**8. Label Encoding vs. One-Hot Encoding?**
Computers only read numbers, not words. 
- **Label Encoding:** Gives words a number rank (Small=0, Medium=1, Large=2). Use this when things have an order.
- **One-Hot Encoding:** Makes a new Yes/No question for each thing (Is it Red? Yes=1, No=0). Use this when no color is "better" than another.

**9. What is Binning? Give an example.**
Binning is putting things into boxes. Instead of remembering every kid's exact age (5, 6, 7...), you put them in boxes labeled "Babies", "Kids", and "Grown-ups". It makes things much easier to understand!

### Answers: Section B: ML Basics

**10. What is Machine Learning?**
Machine Learning is teaching a computer by showing it examples, like a dog learning a trick! Instead of giving the computer exact rules, you just show it lots of pictures of cats until it figures out what a cat looks like on its own.

**11. Supervised vs. Unsupervised Learning?**
- **Supervised Learning:** Like having a teacher give you an answer key to check your work. 
- **Unsupervised Learning:** Like having no teacher. You just sort things into piles that look alike all by yourself.

**12. Regression vs. Classification?**
- **Regression:** Guesses a number, like "How tall will this plant grow?"
- **Classification:** Guesses a category, like "Is this a picture of a dog or a cat?"

**13. Bias vs. Variance?**
- **Bias:** When the computer is stubborn and doesn't learn well, so it makes silly mistakes (too simple).
- **Variance:** When the computer memorizes things too perfectly and gets confused by anything new (too sensitive).

**14. Underfitting vs. Overfitting? How to fix?**
- **Underfitting (High Bias):** Like a student who didn't study at all and fails. Fix it by making the computer study harder (use a smarter model).
- **Overfitting (High Variance):** Like a student who memorized the practice test but fails the real test because the questions changed. Fix it by giving the computer more kinds of practice tests!

**15. Explain the Bias-Variance Tradeoff.**
It's like finding the perfect bed for Goldilocks. You don't want a model that is too simple (underfitting) or too crazy and sensitive (overfitting). You want the "just right" spot in the middle!

### Answers: Section C: Regression

**16. What is Linear Regression and its equation?**
It's drawing a straight line through a bunch of dots on a page to see where they are going. The equation is `y = mx + b`. It helps us guess the future based on the line!

**17. What is a Residual Error?**
It's the distance between our guessed line and the real dot. If we guessed you have 5 apples, but you actually have 7, the error is 2 apples.

**18. What is a Cost Function, and why use MSE?**
A Cost Function is a score of how badly we guessed overall. MSE squares all the errors (makes them positive) and averages them out. We want this bad score to be as small as possible!

**19. How does Gradient Descent work?**
Imagine you are blindfolded on a hill and want to find the bottom. You feel the slope with your feet and take a small step down. You keep stepping down until the ground is flat. That means you found the bottom!

**20. Learning Rate too high vs. too low?**
- **Too high:** You take giant jumps and might jump right over the bottom of the hill!
- **Too low:** You take tiny baby steps and it takes forever to get to the bottom.

**21. Multi-Linear Regression and Multicollinearity?**
Multi-Linear Regression is guessing using many clues instead of one (like guessing toy price using size, color, AND weight). 
Multicollinearity is when two clues are exactly the same and confuse the computer because it doesn't know which one to look at.

**22. What is a Hyperplane?**
In a simple drawing, we use a flat line. In a 3D room, we use a flat piece of paper. A hyperplane is just a fancy word for that flat piece of paper cutting through space!

**23. R² vs Adjusted R²?**
R² is a score (from 0 to 1) of how good our guess is. But if we add silly clues, R² still goes up. Adjusted R² is smarter; it actually lowers our score if we add useless clues.

**24. Polynomial Regression?**
If the dots look like a big smile (a curve), drawing a straight line is silly! Polynomial regression lets us draw a curvy line to fit the smile perfectly.

**25. Regularization? Ridge vs. Lasso?**
Regularization is putting a speed limit on the computer so it doesn't overthink.
- **Ridge:** Slows down all the clues so they aren't too bossy.
- **Lasso:** Actually deletes the useless clues completely!

### Answers: Section D: Classification

**26. What is Logistic Regression?**
Even though it has "regression" in the name, it's a sorter! It guesses if something is A or B (like Spam or Not Spam) by giving a percentage. If it's over 50%, it says YES!

**27. What is the Sigmoid Function?**
It's a magical math slide shaped like an 'S'. It takes any huge or tiny number and squishes it neatly between 0 and 1. This gives us a perfect percentage!

**28. What is a Decision Boundary?**
It's the fence we draw to separate things. On the left side of the fence are cats, and on the right side are dogs!

**29. Linear vs. Logistic Regression?**
Linear guesses a number (like $10). Logistic guesses a group (like Yes or No) by drawing a fence between them.

**30. Why not use MSE for Logistic Regression?**
MSE makes a bumpy hill for Logistic Regression, so the computer gets stuck in the bumps. We use Log Loss instead to make a smooth bowl so the computer can easily slide to the bottom!

**31. What is Entropy in Decision Trees?**
Entropy means a big messy room! If a box has cats and dogs mixed up, it's high entropy. A Decision Tree tries to sort them so one box is just cats and the other is just dogs (perfectly clean!).

**32. How does KNN make a prediction?**
Imagine a new kid comes to school. To see what games they like, we look at their 5 closest friends. If 3 friends like tag and 2 like hide-and-seek, we guess the new kid likes tag!

**33. Euclidean Distance and Scaling for KNN?**
It's just taking a ruler and measuring a straight line between two things. We must scale numbers first, or else giant numbers (like millions of dollars) will completely bully small numbers (like age).

**34. Naive Bayes: Why "Naive"?**
It's a guesser that makes a silly ("naive") rule: it thinks all clues have nothing to do with each other. Even though this is silly, it is surprisingly great at reading text and catching spam emails!

**35. Decision Trees: Information Gain / Gini Index?**
The tree asks Yes/No questions. To pick the best question, it sees which one cleans up the messy room the fastest. This big drop in messiness is called Information Gain.

**36. SVM, Support Vectors, and the Kernel Trick?**
SVM builds a fence between cats and dogs, but it makes the fence as wide as possible! 
- **Support Vectors:** The animals standing closest to the fence.
- **Kernel Trick:** A magic spell that tosses flat dots into the air so we can easily slide a flat sheet between them.

### Answers: Section E: Model Tuning

**37. Parameter vs. Hyperparameter?**
- **Parameter:** Something the computer learns all by itself.
- **Hyperparameter:** A setting YOU have to pick before hitting "Start" (like picking the level on a video game).

**38. What is Cross-Validation?**
Instead of taking just one test and maybe getting lucky, the computer takes 5 different tests on different parts of the data. We average the scores to see how smart it truly is!

**39. Grid Search vs. Random Search?**
- **Grid Search:** Checks every single setting combination one by one (super slow!).
- **Random Search:** Pulls random combinations out of a hat (much faster and usually finds a great one!).

**40. Accuracy vs. Precision vs. Recall (F1/ROC-AUC)?**
Accuracy can lie! If 99 people are healthy and 1 is sick, guessing "Healthy" every time gets 99% accuracy but misses the sick person!
- **Precision:** When we say someone is sick, how often were we right?
- **Recall:** Out of all the sick people, how many did we catch?
- **F1 Score:** A mix of both!

### Answers: Section F: Ensemble Learning

**41. What is Ensemble Learning?**
It's teamwork! Instead of asking one smart kid a hard question, you ask a whole classroom of kids and have them vote. The group vote is almost always better!

**42. Bagging vs. Boosting?**
- **Bagging:** Many kids taking tests at the same time and voting at the end.
- **Boosting:** One kid takes a test, and the next kid specifically tries to fix the first kid's mistakes. They learn in a line!

**43. How does a Random Forest work?**
It's a huge forest of many Decision Trees! To keep them from copying each other, each tree only gets to see a small, random piece of the puzzle. They all vote, and the most votes wins!

**44. AdaBoost vs. Gradient Boosting?**
- **AdaBoost:** Looks at mistakes and makes them flash bright red so the next model pays extra attention.
- **Gradient Boosting:** Looks at the exact math distance of the mistake and tries to guess that distance to subtract it!

**45. What makes XGBoost special?**
It's like Gradient Boosting but wearing a superhero cape. It runs super fast, handles missing pieces automatically, and has built-in rules to stop it from over-memorizing.

### Answers: Section G: Unsupervised Learning

**46. What is Clustering?**
It's sorting a giant pile of Lego blocks. You don't know what to build, but you put the red ones in one pile and the blue ones in another pile just because they look similar.

**47. K-Means algorithm step-by-step & Elbow Method?**
1. Pick how many piles you want (K).
2. Throw K magnets into the data.
3. Every dot sticks to the closest magnet.
4. The magnet moves to the center of its dots.
5. Repeat until they stop moving!
**Elbow Method:** A graph that looks like an arm to help you guess the best number of magnets.

**48. K-Means vs. Hierarchical vs. DBSCAN?**
- **K-Means:** Needs you to guess the number of piles first.
- **Hierarchical:** Builds a giant family tree of dots, and you cut the tree where you want!
- **DBSCAN:** Looks for tight crowds of dots. It can find weird shapes and ignores lonely weirdo dots!

**49. What is PCA and Principal Components?**
If you have a toy with 100 confusing buttons, PCA turns them into 20 super-buttons that do almost the exact same thing. But the super-buttons are confusing to explain because they mash different buttons together!

**50. Anomaly Detection vs. Outlier Removal?**
- **Outlier Removal:** Throwing away broken toys before you start building.
- **Anomaly Detection:** Building a super alarm system that watches for incredibly weird things in the real world, like someone stealing a credit card!