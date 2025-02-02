Below is a simple, self‐contained example that demonstrates how to perform propensity score matching (PSM) on two groups—a study group with 10 patients diagnosed with a condition (here called “stinky brain”) and a control group with 40 patients. In this example, we assume that both groups have the following covariates:

- **age** (numeric)
- **sex** (categorical, e.g., 0 for male, 1 for female)
- **race** (categorical; here encoded as numbers for simplicity)
- **ethnicity** (categorical; also encoded as numbers)

We will use these covariates to estimate the propensity score (the probability of being in the study group) using logistic regression, and then use nearest‐neighbor matching to select controls that best match each treated patient.

> **Prerequisites:**  
> - Python 3  
> - Jupyter Notebook (or another Python environment)  
> - Libraries: `numpy`, `pandas`, `statsmodels`, `sklearn`, and `matplotlib`  
>   
> You can install any missing libraries using `pip install package_name` (for example, `pip install statsmodels`).

---

### Step 1: Import Libraries and Create a Synthetic Dataset

Below, we simulate a dataset for our study group (10 patients with “stinky brain”) and control group (40 patients without the condition). For illustration, we randomly generate covariate values. In a real scenario, you would replace this with your actual data.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

# Create synthetic data for 10 patients with the condition (treatment group)
n_treated = 10
treated = pd.DataFrame({
    'age': np.random.randint(50, 70, n_treated),      # Age between 50 and 70
    'sex': np.random.choice([0, 1], n_treated),         # 0 or 1
    'race': np.random.choice([1, 2, 3], n_treated),       # Coded race categories
    'ethnicity': np.random.choice([0, 1], n_treated),    # 0 or 1 for ethnicity
    'treatment': np.ones(n_treated, dtype=int)           # 1 indicates presence of condition
})

# Create synthetic data for 40 control patients (without the condition)
n_control = 40
control = pd.DataFrame({
    'age': np.random.randint(50, 70, n_control),
    'sex': np.random.choice([0, 1], n_control),
    'race': np.random.choice([1, 2, 3], n_control),
    'ethnicity': np.random.choice([0, 1], n_control),
    'treatment': np.zeros(n_control, dtype=int)         # 0 indicates no condition
})

# Combine the datasets
df = pd.concat([treated, control], ignore_index=True)
print("Dataset head:")
print(df.head())
```

---

### Step 2: Estimate the Propensity Scores

We use logistic regression (from the `statsmodels` package) to model the probability of having “stinky brain” (i.e. being in the treatment group) as a function of age, sex, race, and ethnicity.

```python
# Define the covariates used in the model
covariates = ['age', 'sex', 'race', 'ethnicity']

# Add a constant term to the covariates (required for statsmodels)
X = sm.add_constant(df[covariates])
y = df['treatment']

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)  # disp=False suppresses the fitting output

# Predict propensity scores and add to the DataFrame
df['propensity_score'] = result.predict(X)
print("\nData with Propensity Scores:")
print(df[['age', 'sex', 'race', 'ethnicity', 'treatment', 'propensity_score']].head())
```

---

### Step 3: Perform Nearest-Neighbor Matching

We now match each treated patient to one (or more) control patients based on the similarity of their propensity scores. Here we use nearest-neighbor matching. Although your design is 4:1 matching (four controls per treated), we will show the simple case first (1:1 matching) and then note how you can extend it.

#### **1:1 Matching Example**

```python
# Split data into treated and control groups
treated_df = df[df['treatment'] == 1].copy()
control_df = df[df['treatment'] == 0].copy()

# Initialize Nearest Neighbors algorithm using control propensity scores.
# We need to reshape the propensity scores into a 2D array.
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_df[['propensity_score']])

# For each treated patient, find the nearest control (1:1 matching)
distances, indices = nbrs.kneighbors(treated_df[['propensity_score']])

# Store the index of the matched control in the treated DataFrame
treated_df['match_index'] = control_df.index[indices.flatten()]

# Retrieve matched control observations
matched_controls = df.loc[treated_df['match_index']].copy()

# Combine treated patients with their matched controls
matched_data_1to1 = pd.concat([treated_df, matched_controls]).reset_index(drop=True)
print("\nMatched Data (1:1 Matching) Head:")
print(matched_data_1to1.head())
```

#### **Extending to 4:1 Matching**

For 4:1 matching, we would set `n_neighbors=4` and then choose all four nearest neighbors for each treated patient. One way to combine the matched controls is to keep track of the matching for each treated unit.

```python
# For 4:1 matching: match each treated patient with 4 nearest controls
nbrs_4 = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(control_df[['propensity_score']])
distances_4, indices_4 = nbrs_4.kneighbors(treated_df[['propensity_score']])

# Create a DataFrame to store matching pairs: each treated patient will appear 4 times
matched_pairs = []

# Loop through each treated patient and record the treated index and each control index
for i, treated_index in enumerate(treated_df.index):
    # Get indices of the 4 matched controls from control_df (indices_4[i] are positions in control_df)
    control_indices = control_df.index[indices_4[i]]
    for ci in control_indices:
        matched_pairs.append({
            'treated_index': treated_index,
            'control_index': ci,
            'treated_propensity': treated_df.loc[treated_index, 'propensity_score'],
            'control_propensity': df.loc[ci, 'propensity_score']
        })

matched_pairs_df = pd.DataFrame(matched_pairs)
print("\nMatched Pairs (4:1 Matching):")
print(matched_pairs_df.head(8))
```

*Note:* In 4:1 matching you now have multiple control observations for each treated unit. In further analysis (for example, estimating treatment effects), you may need to account for the fact that controls are repeated. Various approaches exist (e.g., weighting, clustering standard errors) depending on your analysis.

---

### Step 4: Inspect and Plot the Matched Data

After matching, it’s important to check if the covariate distributions between treated and matched control groups are similar.

```python
# For the 1:1 matched data:
print("\nSummary Statistics for 1:1 Matched Data:")
print(matched_data_1to1.groupby('treatment')[['age', 'sex', 'race', 'ethnicity', 'propensity_score']].describe())

# Plot propensity score distributions for the matched groups
plt.figure(figsize=(8, 4))
plt.hist(matched_data_1to1[matched_data_1to1['treatment'] == 1]['propensity_score'], bins=10, alpha=0.5, label='Treated')
plt.hist(matched_data_1to1[matched_data_1to1['treatment'] == 0]['propensity_score'], bins=10, alpha=0.5, label='Control')
plt.xlabel('Propensity Score')
plt.ylabel('Frequency')
plt.title('Propensity Score Distribution (1:1 Matched)')
plt.legend()
plt.show()
```

---

### Recap

1. **Data Preparation:**  
   - We created synthetic data for 10 treated patients (with “stinky brain”) and 40 control patients, including covariates such as age, sex, race, and ethnicity.

2. **Propensity Score Estimation:**  
   - We used logistic regression to estimate the propensity score (probability of being in the treated group) based on the covariates.

3. **Matching:**  
   - We performed nearest-neighbor matching (first in a simple 1:1 case, then extended the logic to 4:1 matching) to pair treated patients with similar control patients.
   
4. **Evaluation:**  
   - We inspected the matched data by summarizing covariate distributions and plotting the propensity score distributions to assess the balance between groups.

You can run these code snippets cell-by-cell in your Jupyter Notebook to recreate the analysis. Adjust and extend the example based on your real data and research questions.