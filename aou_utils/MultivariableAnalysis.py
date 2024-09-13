import pandas as pd
import numpy as np
import statsmodels.api as sm
import re

class MultivariableAnalysis:
    def __init__(self, study_group_df, control_group_df):
        """
        Initialize the MultivariableAnalysis class.

        Parameters:
        study_group_df (pd.DataFrame): The DataFrame for the study group.
        control_group_df (pd.DataFrame): The DataFrame for the control group.
        """
        # Make copies of the dataframes to prevent unintended modifications
        self.combined_df = pd.concat([study_group_df.copy(), control_group_df.copy()])
        self.study_group_df = study_group_df.copy()
        self.independent_vars = []

    def add_flag(self, flag_var):
        """
        Add a flag variable to indicate the study group status.

        Parameters:
        flag_var (str): The name of the flag variable to add, which will indicate whether a person belongs to the study group.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        self.combined_df[flag_var] = self.combined_df['person_id'].isin(self.study_group_df['person_id']).astype(int)
        self.independent_vars.append(flag_var)
        return self

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

    def add_dummies(self, column_name, drop_category=None):
        """
        Convert a categorical variable into dummy variables for a specified column,
        and drop a specified category or the first category if none is specified.

        The function will:
        - Create dummy variables for each category of the specified column.
        - Optionally drop one category (specified or the first one by default) to avoid the dummy variable trap.
        - Transform column names to be lowercase, replace spaces with underscores, and remove non-alphanumeric characters.
        
        Parameters:
        column_name (str): The name of the categorical column to convert into dummy variables.
        drop_category (str, optional): The category to drop, which will serve as the reference group. If not specified, the first category will be dropped.

        Returns:
        MultivariableAnalysis1: Returns the instance to allow method chaining.

        Raises:
        Warning: If the specified `drop_category` does not exist in the dummy columns.
        """
        # Generate dummy variables and apply name cleaning
        dummies = pd.get_dummies(self.combined_df[column_name], prefix=self._clean_column_name(column_name), dtype=int)

        if drop_category is None:
            # Use pandas built-in method to drop the first category if drop_category is not specified
            dummies = pd.get_dummies(self.combined_df[column_name], prefix=self._clean_column_name(column_name), drop_first=True, dtype=int)
        else:
            # Apply the same transformation to drop_category to match the modified column names
            drop_category_cleaned = self._clean_column_name(f"{column_name}_{drop_category}")
            
            # Check if the drop_category exists in the columns of the dummies
            if drop_category_cleaned in dummies.columns:
                # Drop the specified category
                dummies.drop(columns=[drop_category_cleaned], inplace=True)
            else:
                print(f"Warning: The category '{drop_category}' does not exist in the column '{column_name}'. No category will be dropped.")

        # Concatenate the dummy variables to the combined DataFrame
        self.combined_df = pd.concat([self.combined_df, dummies], axis=1)

        # Add the dummy variable column names to the list of independent variables
        self.independent_vars.extend(dummies.columns)

        return self

    def add_independent_var(self, variable_name):
        """
        Add an independent variable to the model.

        Parameters:
        variable_name (str): The name of the independent variable to add.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        self.independent_vars.append(variable_name)
        return self

    def ensure_numeric(self, column):
        """
        Ensure that a column is numeric, and attempt to convert it if necessary.

        Parameters:
        column (str): The name of the column to check and convert.

        Returns:
        None
        """
        if not pd.api.types.is_numeric_dtype(self.combined_df[column]):
            print(f"Warning: The column '{column}' is not in numerical form. Attempting to convert.")
            self.combined_df[column] = pd.to_numeric(self.combined_df[column], errors='coerce')
            if self.combined_df[column].isna().any():
                print(f"Warning: Conversion of column '{column}' resulted in NaNs. These rows will be dropped.")

    def fit_model(self, dependent_var):
        """
        Fit a logistic regression model using the specified dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Ensure all relevant columns are numeric
        for var in self.independent_vars + [dependent_var]:
            self.ensure_numeric(var)

        # Drop rows with NaN values that might have resulted from the coercion
        self.combined_df.dropna(subset=self.independent_vars + [dependent_var], inplace=True)

        # Verify data types
        for var in self.independent_vars + [dependent_var]:
            print(f"Data type of '{var}': {self.combined_df[var].dtype}")

        # Define the independent variables (X) and dependent variable (y)
        X = sm.add_constant(self.combined_df[self.independent_vars])
        y = self.combined_df[dependent_var]

        # Check shapes of X and y
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

        # Fit the logistic regression model
        self.model = sm.Logit(y, X).fit()
        return self

    def convert_column_to_type(self, column_name, dtype):
        """
        Convert a column to a specified data type.

        Parameters:
        column_name (str): The name of the column to convert.
        dtype (str): The target data type.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        try:
            self.combined_df[column_name] = self.combined_df[column_name].astype(dtype)
            print(f"Successfully converted column '{column_name}' to {dtype}.")
        except Exception as e:
            print(f"Error converting column '{column_name}' to {dtype}: {e}")
        return self
    
    @staticmethod
    def format_p_value(p_value):
        """
        Format the p-value for display purposes.

        Parameters:
        p_value (float): The p-value to format.

        Returns:
        str: The formatted p-value as a string.
        """
        if p_value < 0.001:
            return "<0.001"
        else:
            return f"{p_value:.3f}"
    
    def get_results(self):
        """
        Retrieve and display the results of the logistic regression analysis.

        Returns:
        pd.DataFrame: A DataFrame containing the odds ratios, confidence intervals, and formatted results.
        """
        # Get the odds ratios and confidence intervals
        odds_ratios = self.model.params
        confidence_intervals = self.model.conf_int()

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Variable': self.model.params.index,
            'Odds Ratio': np.exp(odds_ratios),
            'Lower CI': np.exp(confidence_intervals.iloc[:, 0]),
            'Upper CI': np.exp(confidence_intervals.iloc[:, 1]),
            'P-value': self.model.pvalues
        })
        
        # Format the p-values
        results_df['P-value'] = results_df['P-value'].apply(self.format_p_value)

        # Round the odds ratios and confidence intervals to 2 decimal places
        results_df[['Odds Ratio', 'Lower CI', 'Upper CI']] = results_df[['Odds Ratio', 'Lower CI', 'Upper CI']].round(2)

        # Create a new column with the formatted results
        results_df['Formatted Results'] = results_df.apply(lambda row: f"{row['Odds Ratio']} ({row['Lower CI']}-{row['Upper CI']})", axis=1)

        # Print the results DataFrame
        print(results_df)

        return results_df
