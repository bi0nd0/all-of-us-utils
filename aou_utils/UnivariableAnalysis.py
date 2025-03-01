import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import fisher_exact
import re


class UnivariableAnalysis:
    def __init__(self, study_group_df, control_group_df):
        """
        Initialize the UnivariableAnalysis class.

        Parameters:
        study_group_df (pd.DataFrame): The DataFrame for the study group.
        control_group_df (pd.DataFrame): The DataFrame for the control group.
        """
        # Make copies of the dataframes to prevent unintended modifications
        self.combined_df = pd.concat([study_group_df.copy(), control_group_df.copy()])
        self.study_group_df = study_group_df.copy()
        self.independent_vars = []
        self.debug = False

    def add_flag(self, flag_var):
        """
        Add a flag variable to indicate the study group status.

        Parameters:
        flag_var (str): The name of the flag variable.

        Returns:
        UnivariableAnalysis: Returns the instance to allow method chaining.
        """
        self.combined_df[flag_var] = self.combined_df['person_id'].isin(self.study_group_df['person_id']).astype(int)
        return self
    
    def add_indipendent_var(self, column):
        self.independent_vars.append(column)
    
    @staticmethod
    def format_p_value(p_value):
        """
        Format the p-value for display.

        Parameters:
        p_value (float): The p-value to format.

        Returns:
        str: The formatted p-value.
        """
        if p_value < 0.001:
            return "<0.001"
        else:
            return f"{p_value:.3f}"

    def run_analysis(self, dependent_var, independent_var):
        """
        Run the univariable logistic regression analysis.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        independent_var (str): The name of the independent variable.

        Returns:
        pd.DataFrame: A DataFrame with the analysis results.
        """
        # Debugging: Print the first few rows of the DataFrame to ensure columns are present
        if self.debug:
            print("Combined DataFrame head:\n", self.combined_df.head())
            print(f"Checking if '{independent_var}' is in combined_df columns: {independent_var in self.combined_df.columns}")
            print(f"Checking if '{dependent_var}' is in combined_df columns: {dependent_var in self.combined_df.columns}")

        if independent_var not in self.combined_df.columns or dependent_var not in self.combined_df.columns:
            raise ValueError(f"One or both of the specified columns '{dependent_var}' or '{independent_var}' are not present in the DataFrame.")

        # Define the dependent variable (y) and independent variable (x)
        y = self.combined_df[dependent_var]
        x = self.combined_df[[independent_var]]

        # Check for perfect separation
        contingency_table = pd.crosstab(self.combined_df[independent_var], self.combined_df[dependent_var])
        if self.debug:
            print("\nContingency Table:")
            print(contingency_table)

        if (contingency_table == 0).values.any():
            perform_fisher = True
        else:
            perform_fisher = False

        if perform_fisher:
            odds_ratio, p_value = fisher_exact(contingency_table)
            formatted_p_value = self.format_p_value(p_value)
            results_df = pd.DataFrame({
                'Variable': [independent_var],
                'Odds Ratio': [odds_ratio],
                'Lower CI': [np.nan],  # Fisher's Exact Test does not provide CI by default
                'Upper CI': [np.nan],
                'P-value': [p_value]
            })
            print(f"Univariable analysis for {independent_var} using Fisher's Exact Test: OR = {odds_ratio:.2f}, p ≈ {formatted_p_value}")
        else:
            # Add a constant (intercept) to the independent variable
            x = sm.add_constant(x)

            # Fit the logistic regression model
            model = sm.Logit(y, x).fit()

            # Debugging: Print the model summary to check the coefficients
            if self.debug:
                print(model.summary())

            # Get the odds ratio and 95% CI using the variable name instead of index
            odds_ratio = np.exp(model.params[independent_var])
            confidence_interval = np.exp(model.conf_int().loc[independent_var])

            # Get the p-value using the variable name instead of index
            p_value = model.pvalues[independent_var]

            # Format the p-value
            formatted_p_value = self.format_p_value(p_value)

            # Create a DataFrame with the results
            results_df = pd.DataFrame({
                'Variable': [independent_var],
                'Odds Ratio': [odds_ratio],
                'Lower CI': [confidence_interval[0]],
                'Upper CI': [confidence_interval[1]],
                'P-value': [p_value]
            })

            # Print the results
            print(f"Univariable analysis for {independent_var}: OR = {odds_ratio:.2f} ({confidence_interval[0]:.2f}-{confidence_interval[1]:.2f}), p ≈ {formatted_p_value}")

        return results_df

    
    def convert_column_to_type(self, column_name, dtype):
        """
        Convert a specified column to a specified type.

        Parameters:
        column_name (str): The name of the column to convert.
        dtype (str): The data type to convert the column to.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        try:
            self.combined_df[column_name] = self.combined_df[column_name].astype(dtype)
            if self.debug:
                print(f"Successfully converted column '{column_name}' to {dtype}.")
        except Exception as e:
            print(f"Error converting column '{column_name}' to {dtype}: {e}")
        return self

    import re

    @staticmethod
    def _clean_column_name(column_name):
        """
        Clean column names by:
        - Lowercasing
        - Replacing spaces with underscores
        - Removing non-alphanumeric characters
        
        Parameters:
        column_name (str): The column name to clean.
        
        Returns:
        str: The cleaned column name.
        """
        # Lowercase, replace spaces, and remove non-alphanumeric characters in one step
        return re.sub(r'\W+', '', column_name.lower().replace(' ', '_'))

    def add_dummies(self, column_name):
        """
        Convert a categorical variable into dummy variables for a specified column,
        and drop a specified category or the first category if none is specified.

        The function will:
        - Create dummy variables for each category of the specified column.
        - Transform column names to be lowercase, replace spaces with underscores, and remove non-alphanumeric characters.
        
        Parameters:
        column_name (str): The name of the categorical column to convert into dummy variables.
        
        Returns:
        UnivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Generate dummy variables and apply name cleaning
        dummies = pd.get_dummies(self.combined_df[column_name], prefix=self._clean_column_name(column_name), drop_first=False, dtype=int)

        # Concatenate the dummy variables to the combined DataFrame
        self.combined_df = pd.concat([self.combined_df, dummies], axis=1)

        # Add the dummy variable column names to the list of independent variables
        self.independent_vars.extend(dummies.columns)

        return self
    
    def setDebug(self, flag: bool=True):
        self.debug = flag
        return self
    
    def run_all_analyses(self, dependent_var):
        """
        Run univariable logistic regression for all independent variables against a specified dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        pd.DataFrame: A DataFrame with the combined results of all univariable analyses.
        """
        all_results = pd.DataFrame()  # Initialize an empty DataFrame to store all results
        
        # Loop through all independent variables and run the analysis
        for independent_var in self.independent_vars:
            if self.debug:
                print(f"Running analysis for {independent_var} against {dependent_var}")
            result = self.run_analysis(dependent_var, independent_var)
            
            # Append the result of the current analysis to the all_results DataFrame
            all_results = pd.concat([all_results, result], ignore_index=True)
        
        return all_results
