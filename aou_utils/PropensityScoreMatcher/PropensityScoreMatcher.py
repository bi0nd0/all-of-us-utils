import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class PropensityScoreMatcher:
    def __init__(self, study_df: pd.DataFrame, control_df: pd.DataFrame):
        """
        Initialize with study (treated) and control DataFrames.
        Both dataframes must contain the same columns for the covariates.
        """
        self.study_df = study_df.copy()
        self.control_df = control_df.copy()
        self.covariates = []          # list of covariate column names
        self.num_neighbors = 1        # default number of neighbors for matching
        self.results = None           # placeholder for results after run

    def add_covariate(self, covariate: str):
        """Builder method to add a covariate (one at a time)."""
        if covariate not in self.covariates:
            self.covariates.append(covariate)
        return self

    def set_covariates(self, covariate_list: list):
        """Builder method to set all covariates at once."""
        self.covariates = covariate_list
        return self

    def set_num_neighbors(self, num_neighbors: int):
        """Builder method to set the number of neighbors for matching."""
        self.num_neighbors = num_neighbors
        return self

    def run(self):
        """
        Run the propensity score estimation and matching.
        Returns a dictionary with keys: 
          'model': the fitted logistic regression model result,
          'combined_df': the DataFrame with both treated and matched controls,
          'matched_pairs': if num_neighbors > 1, the matching pairs info.
        """
        # Combine study and control into one DataFrame.
        study = self.study_df.copy()
        control = self.control_df.copy()
        study['treatment'] = 1  # ensure treatment indicator is 1 for study
        control['treatment'] = 0  # ensure treatment indicator is 0 for control
        combined_df = pd.concat([study, control], ignore_index=True)
        
        if not self.covariates:
            raise ValueError("No covariates specified. Please add at least one covariate.")
        
        # Estimate propensity scores using logistic regression.
        X = combined_df[self.covariates]
        X = sm.add_constant(X)
        y = combined_df['treatment']
        logit_model = sm.Logit(y, X)
        model_result = logit_model.fit(disp=False)
        combined_df['propensity_score'] = model_result.predict(X)

        # Split the data back into treated and control groups.
        treated_df = combined_df[combined_df['treatment'] == 1].copy()
        control_df = combined_df[combined_df['treatment'] == 0].copy()

        # Matching using nearest neighbors based on the propensity score.
        nbrs = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='ball_tree')
        nbrs.fit(control_df[['propensity_score']])
        distances, indices = nbrs.kneighbors(treated_df[['propensity_score']])
        
        # If only one neighbor per treated unit, do a simple 1:1 matching.
        if self.num_neighbors == 1:
            treated_df['match_index'] = control_df.index[indices.flatten()]
            matched_controls = combined_df.loc[treated_df['match_index']].copy()
            matched_df = pd.concat([treated_df, matched_controls]).reset_index(drop=True)
            match_info = None
        else:
            # For multiple neighbors, store matching pairs.
            match_pairs = []
            for i, treated_idx in enumerate(treated_df.index):
                control_indices = control_df.index[indices[i]]
                for ci in control_indices:
                    match_pairs.append({
                        'treated_index': treated_idx,
                        'control_index': ci,
                        'treated_propensity': treated_df.loc[treated_idx, 'propensity_score'],
                        'control_propensity': combined_df.loc[ci, 'propensity_score']
                    })
            match_info = pd.DataFrame(match_pairs)
            # For display, you might combine all treated and control observations.
            # Here, we simply return the matching pairs.
            matched_df = None

        self.results = {
            'model': model_result,
            'combined_df': combined_df,
            'matched_df': matched_df,
            'matched_pairs': match_info
        }
        return self.results

    @staticmethod
    def display_results(results: dict):
        """
        Display results from the run method.
        If 1:1 matching was done, it displays the combined matched DataFrame summary.
        Otherwise, it displays the matching pairs.
        """
        model_result = results.get('model')
        combined_df = results.get('combined_df')
        matched_df = results.get('matched_df')
        matched_pairs = results.get('matched_pairs')
        
        print("Logistic Regression Model Summary:")
        print(model_result.summary())
        print("\nCombined DataFrame Head (with propensity scores):")
        print(combined_df.head())
        
        if matched_df is not None:
            print("\nMatched Data (1:1 matching):")
            print(matched_df.head())
            print("\nSummary statistics by treatment group:")
            print(matched_df.groupby('treatment').describe())
            
            # Optional: plot propensity score distributions.
            plt.figure(figsize=(8, 4))
            plt.hist(matched_df[matched_df['treatment'] == 1]['propensity_score'], bins=10, 
                     alpha=0.5, label='Treated')
            plt.hist(matched_df[matched_df['treatment'] == 0]['propensity_score'], bins=10, 
                     alpha=0.5, label='Control')
            plt.xlabel('Propensity Score')
            plt.ylabel('Frequency')
            plt.title('Propensity Score Distribution (1:1 Matched)')
            plt.legend()
            plt.show()
        elif matched_pairs is not None:
            print("\nMatched Pairs (Multiple Neighbors):")
            print(matched_pairs.head())
        else:
            print("No matching results available.")