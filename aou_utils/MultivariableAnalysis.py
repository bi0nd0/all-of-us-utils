import pandas as pd
import numpy as np
import statsmodels.api as sm

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
        flag_var (str): The name of the flag variable.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        self.combined_df[flag_var] = self.combined_df['person_id'].isin(self.study_group_df['person_id']).astype(int)
        self.independent_vars.append(flag_var)
        return self

    def convert_categorical_to_dummies(self, column_name, drop_category=None):
        """
        Convert categorical variables to dummy variables for a specified column,
        and drop a specified category or the first category if none is specified.

        Parameters:
        column_name (str): The name of the column to convert.
        drop_category (str, optional): The category to drop. Defaults to None.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Generate dummy variables without dropping any category initially
        dummies = pd.get_dummies(self.combined_df[column_name])
        
        if drop_category is None:
            # Drop the first category if drop_category is not specified
            drop_category = dummies.columns[0]
        
        # Check if the drop_category exists in the columns of the dummies
        if drop_category in dummies.columns:
            # Drop the specified category
            dummies.drop(columns=[drop_category], inplace=True)
        else:
            print(f"Warning: The category '{drop_category}' does not exist in the column '{column_name}'. No category will be dropped.")
        
        # Convert dummy variables to integers
        dummies = dummies.astype(int)
        # Concatenate the dummy variables to the combined DataFrame
        self.combined_df = pd.concat([self.combined_df, dummies], axis=1)
        # Add the dummy variable column names to the list of independent variables
        self.independent_vars.extend(dummies.columns)
        return self


    def add_independent_var(self, variable_name):
        """
        Add an independent variable for the analysis.

        Parameters:
        variable_name (str): The name of the independent variable to add.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        self.independent_vars.append(variable_name)
        return self

    def ensure_numeric(self, column):
        """
        Ensure a column is numeric, and provide a warning if conversion is necessary.

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
        Fit the logistic regression model.

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
        Convert a specified column to a specified type.

        Parameters:
        column_name (str): The name of the column to convert.
        dtype (str): The data type to convert the column to.

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
    
    def get_results(self):
        """
        Get the results of the logistic regression analysis.

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