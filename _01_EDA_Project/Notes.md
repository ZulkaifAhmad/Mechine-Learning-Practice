# Preprocessing Methods Cheatsheet

## Definitions
- **Data Cleaning** — fixing missing/incorrect/inconsistent data (no new info added)
- **Feature Engineering** — creating or transforming columns to add useful signal (encoding, binning, extraction)
- **Label Encoding** — mapping categories to numbers in the *same* column (best for 2 categories)
- **One-Hot Encoding** — splitting a category column into multiple 0/1 columns (best for 3+ categories)
- **Binning** — grouping a *numeric* column into ranges (bins) and giving each range a label. This label is usually text (e.g. "Young", "Middle", "Old"), so binning actually flips the normal direction: instead of converting text → number (like encoding), it converts number → text/category. That's why binned columns almost always need Label/One-Hot Encoding done on them afterward, if you want them numeric again for the model.
- **Feature Selection** — keeping only the columns that matter, dropping the rest
- **Scaling** — adjusting numeric range only, meaning of the column stays the same. Important: scaling is only applied to **input/feature columns (X)** — the numbers you feed into the model. It should **never** be applied to the **output/target column (y)** (here, `HeartDisease`), because the target needs to stay in its original, meaningful scale (e.g. 0/1 for classification). So in the code below, `HeartDisease` is deliberately left out of the `StandardScaler` call.
- **Preprocessing** — the umbrella term covering all of the above, done before model training

## Sample DataFrame (Heart Disease dataset)
| Age | Sex | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
|-----|-----|----------------|-----------|-------------|-----------|------------|-------|----------------|---------|----------|--------------|
| 40  | M   | ATA            | 140       | 289         | 0         | Normal     | 172   | N              | 0       | Up       | 0            |
| 49  | F   | NAP            | 160       | 180         | 0         | Normal     | 156   | N              | 1       | Flat     | 1            |
| 37  | M   | ATA            | 130       | 283         | 0         | ST         | 98    | N              | 0       | Up       | 0            |
| 48  | F   | ASY            | 138       | 214         | 0         | Normal     | 108   | Y              | 1.5     | Flat     | 1            |

## Methods applied to this dataset

```python
df = df.dropna(subset=['Cholesterol'])                    
# Data Cleaning

df['Sex'] = df['Sex'].map({'M':1,'F':0})                   
# Feature Engineering - Label Encoding (2 categories)

df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y':1,'N':0})  
# Feature Engineering - Label Encoding (2 categories)

df = pd.get_dummies(df, columns=['ChestPainType'])          
# Feature Engineering - One-Hot Encoding (4 categories: ATA,NAP,ASY,TA)

df = pd.get_dummies(df, columns=['RestingECG'])              
# Feature Engineering - One-Hot Encoding (3 categories: Normal,ST,LVH)

df = pd.get_dummies(df, columns=['ST_Slope'])                
# Feature Engineering - One-Hot Encoding (3 categories: Up,Flat,Down)

# Feature Engineering - Binning
# Binning takes a NUMERIC column (Age) and converts it into a TEXT/category column
# by grouping values into ranges and giving each range a label (bin name).
# So here, numeric Age values like 37, 40, 48, 49 get turned into text labels
# like "Young", "Middle", "Old" - this is the opposite direction of encoding.
df['AgeGroup'] = pd.cut(df['Age'],
                         bins=[0, 40, 55, 100],
                         labels=['Young', 'Middle', 'Old'])    
                        # Feature Engineering - Binning (numeric -> text)

# Since AgeGroup is now text (categorical), if we still want it numeric for the model,
# we must encode it again - here it has 3 categories so One-Hot Encoding fits:
df = pd.get_dummies(df, columns=['AgeGroup'])                  
# Feature Engineering - One-Hot Encoding (after binning)

df = df.drop(columns=['FastingBS'])                            
# Feature Selection (example - low predictive value)

from sklearn.preprocessing import StandardScaler
df[['Age','RestingBP','Cholesterol','MaxHR']] = StandardScaler().fit_transform(
    df[['Age','RestingBP','Cholesterol','MaxHR']])
    
# Scaling - applied ONLY to input/feature columns (Age, RestingBP, Cholesterol, MaxHR)
# 'HeartDisease' (the target/output column) is NOT scaled - it must stay as 0/1
```

## What else you should add (missing from this sheet)

1. **Train-Test Split, done BEFORE scaling** — split first with `train_test_split()`, then `fit_transform()` the scaler on training data only and `transform()` (not fit) on test data. Fitting the scaler on the whole dataset (like this sheet currently does) causes **data leakage** — the model indirectly "sees" test data statistics before evaluation.
2. **Missing value imputation** — this sheet only shows `dropna()` (deleting rows). Often you instead *fill* missing values so you don't lose data:
   ```python
   df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())   # Data Cleaning - Imputation
   ```
3. **Ordinal Encoding** — for categories that DO have order (unlike `ChestPainType`, which has no order). Example: if you had a `Severity` column with Low/Medium/High, you'd map it to 0/1/2 to preserve the order — different from Label Encoding, which is only for 2-category unordered columns:
   ```python
   df['Severity'] = df['Severity'].map({'Low':0,'Medium':1,'High':2})   # Feature Engineering - Ordinal Encoding
   ```
4. **Outlier handling** — extreme values (e.g. Cholesterol = 0 in real heart disease data is a data error, not a real reading) can distort scaling and model training. Common approach: IQR method or capping.
5. **Duplicate removal** — `df.drop_duplicates()`, part of Data Cleaning, not shown yet.
6. **Normalization vs Standardization** — this sheet uses `StandardScaler` (mean=0, std=1). `MinMaxScaler` (scales to a 0–1 range) is the other common option — worth knowing both exist and when to pick each.
7. **Data type conversion** — e.g. making sure numeric-looking columns aren't accidentally stored as text (`df['Age'] = df['Age'].astype(int)`).

## Quick rule
- **2 categories, 0/1 meaning** → Label Encoding (`.map()`) → used on `Sex`, `ExerciseAngina`
- **3+ categories, no order** → One-Hot Encoding (`get_dummies()`) → used on `ChestPainType`, `RestingECG`, `ST_Slope`
- **Numeric column → grouped ranges → text labels** → Binning (`pd.cut()`) → used on `Age` → `AgeGroup`
- **Text/category result from binning still needs encoding** → apply Label/One-Hot Encoding *after* binning if you want it numeric again
- **Only changes number range, meaning unchanged** → Scaling → used on `Age`, `RestingBP`, `Cholesterol`, `MaxHR`
- **Everything above combined** → Preprocessing