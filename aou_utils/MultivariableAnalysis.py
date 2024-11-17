import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import warnings

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
        self.debug = False

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
        dummies = pd.get_dummies(self.combined_df[column_name], prefix=column_name, drop_first=False, dtype=int)
              
        # Ensure both the dummy variable column names and the drop category are treated consistently (lowercased)
        dummies.columns = [self._clean_column_name(col) for col in dummies.columns]
        
        if self.debug:
            print(dummies.columns)
        
        # If no drop_category is specified, drop the first column automatically
        if drop_category is not None:
            # Clean the drop_category to match the dummy column names
            drop_category_cleaned = self._clean_column_name(f"{column_name}_{drop_category}")
            print(drop_category_cleaned)
            # Check if the cleaned category exists in the dummy columns and drop it if present
            if drop_category_cleaned in dummies.columns:
                dummies.drop(columns=[drop_category_cleaned], inplace=True)
            else:
                print(f"Warning: The category '{drop_category}' does not exist in the column '{column_name}'. No category will be dropped.")
        else:
            # If no drop_category is specified, drop the first column automatically
            dummies = dummies.iloc[:, 1:]

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

    def check_perfect_separation(self, dependent_var):
        """
        Check for perfect separation between independent variables and the dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        None
        """
        for var in self.independent_vars:
            if var in self.combined_df.columns:
                crosstab = pd.crosstab(self.combined_df[var], self.combined_df[dependent_var])
                if self.debug:
                    print(f"\nCrosstab of {var} and {dependent_var}:")
                    print(crosstab)
                # Check for categories where the outcome is always the same
                if (crosstab[0] == 0).any() or (crosstab[1] == 0).any():
                    print(f"Warning: Variable '{var}' may cause perfect separation.")

    def check_multicollinearity(self):
        """
        Calculate VIF for each independent variable to detect multicollinearity.
        Handles cases where division by zero occurs due to perfect multicollinearity by suppressing RuntimeWarnings.

        Returns:
        pd.DataFrame: A DataFrame containing VIF values.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import warnings

        X = self.combined_df[self.independent_vars]
        X = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns

        # Initialize an empty list to store VIF values
        vif_values = []

        # Calculate VIF for each variable
        for i in range(X.shape[1]):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_values.append(vif)
                except (np.linalg.LinAlgError, ZeroDivisionError):
                    # Handle cases where division by zero occurs
                    vif_values.append(np.inf)

        vif_data["VIF"] = vif_values

        if self.debug:
            print("\nVariance Inflation Factor (VIF):")
            print(vif_data)
        return vif_data



    def check_data_quality(self, dependent_var):
        """
        Check data types, missing values, and variables with zero variance.

        Parameters:
        dependent_var (str): The name of the dependent variable.

        Returns:
        None
        """
        print("\nData Types:")
        print(self.combined_df[self.independent_vars + [dependent_var]].dtypes)

        print("\nMissing Values:")
        print(self.combined_df[self.independent_vars + [dependent_var]].isnull().sum())

        print("\nVariables with Zero Variance:")
        for var in self.independent_vars:
            if self.combined_df[var].nunique() <= 1:
                print(f"Variable '{var}' has zero variance.")

    def plot_correlation_matrix(self):
        """
        Plot a heatmap of the correlation matrix of independent variables.

        Returns:
        None
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        X = self.combined_df[self.independent_vars]
        corr_matrix = X.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix of Independent Variables")
        plt.show()

    def fit_model_with_regularization(self, dependent_var, method='l1', alpha=0.1):
        """
        Fit a logistic regression model using regularization.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        method (str): Regularization method ('l1' or 'l2').
        alpha (float): Regularization strength.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Ensure all relevant columns are numeric
        for var in self.independent_vars + [dependent_var]:
            self.ensure_numeric(var)

        # Drop rows with NaN values
        self.combined_df.dropna(subset=self.independent_vars + [dependent_var], inplace=True)

        # Define the independent variables (X) and dependent variable (y)
        X = sm.add_constant(self.combined_df[self.independent_vars])
        y = self.combined_df[dependent_var]

        # Fit the logistic regression model with regularization
        try:
            self.model = sm.Logit(y, X).fit_regularized(method=method, alpha=alpha, maxiter=1000)
        except Exception as e:
            print(f"Error fitting model with regularization: {e}")
        return self

    def fit_model(self, dependent_var, robust=False):
        """
        Fit a logistic regression model using the specified dependent variable.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        robust (bool): Whether to use robust standard errors.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Ensure all relevant columns are numeric
        for var in self.independent_vars + [dependent_var]:
            self.ensure_numeric(var)

        # Drop rows with NaN values
        self.combined_df.dropna(subset=self.independent_vars + [dependent_var], inplace=True)

        # Define the independent variables (X) and dependent variable (y)
        X = sm.add_constant(self.combined_df[self.independent_vars])
        y = self.combined_df[dependent_var]

        # Fit the logistic regression model
        try:
            if robust:
                self.model = sm.Logit(y, X).fit(cov_type='HC3', maxiter=1000)
            else:
                self.model = sm.Logit(y, X).fit(maxiter=1000)
        except Exception as e:
            print(f"Error fitting model: {e}")
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
            if self.debug:
                print(f"Successfully converted column '{column_name}' to {dtype}.")
        except Exception as e:
            if self.debug:
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
    

    def setDebug(self, flag: bool=True):
        self.debug = flag
        return self

    def automate_analysis(self, dependent_var, vif_threshold=5.0, regularization_method='l1', alpha=0.1):
        """
        Automate the analysis by performing diagnostics, adjusting variables, and fitting the model.

        Parameters:
        dependent_var (str): The name of the dependent variable.
        vif_threshold (float): Threshold for VIF to detect multicollinearity. Variables with VIF above this value will be removed.
        regularization_method (str): Regularization method to use if needed ('l1' for Lasso, 'l2' for Ridge).
        alpha (float): Regularization strength.

        Returns:
        MultivariableAnalysis: Returns the instance to allow method chaining.
        """
        # Step 1: Check data quality
        if self.debug:
            print("Checking data quality...")
            self.check_data_quality(dependent_var)

        # Step 2: Check for perfect separation and remove problematic variables
        if self.debug:
            print("\nChecking for perfect separation...")
        vars_to_remove = []
        for var in self.independent_vars:
            if var in self.combined_df.columns:
                crosstab = pd.crosstab(self.combined_df[var], self.combined_df[dependent_var])
                # Check if any category perfectly predicts the outcome
                zero_in_column = (crosstab == 0).any(axis=1)
                if zero_in_column.any():
                    if self.debug:
                        print(f"Variable '{var}' may cause perfect separation and will be removed.")
                    vars_to_remove.append(var)
        # Remove variables causing perfect separation
        for var in vars_to_remove:
            self.independent_vars.remove(var)

        # Step 3: Check for multicollinearity and remove variables with high VIF
        if self.debug:
            print("\nChecking for multicollinearity...")
        while True:
            vif_data = self.check_multicollinearity()
            vif_data = vif_data[vif_data['Variable'] != 'const']  # Exclude constant term
            max_vif = vif_data['VIF'].max()
            if max_vif > vif_threshold:
                # Remove the variable with the highest VIF
                max_vif_var = vif_data.loc[vif_data['VIF'] == max_vif, 'Variable'].values[0]
                if self.debug:
                    print(f"Variable '{max_vif_var}' has VIF={max_vif:.2f} and will be removed.")
                self.independent_vars.remove(max_vif_var)
            else:
                if self.debug:
                    print("No multicollinearity issues detected.")
                break

        # Step 4: Fit the model
        if self.debug:
            print("\nFitting the model...")
        try:
            # First attempt without regularization
            self.fit_model(dependent_var)
        except Exception as e:
            print(f"Error fitting model: {e}")
            print("Attempting to fit model with regularization.")
            try:
                # Attempt to fit model with regularization as a fallback
                self.fit_model_with_regularization(dependent_var, method=regularization_method, alpha=alpha)
            except Exception as reg_error:
                print(f"Error fitting model with regularization: {reg_error}")
                print("Consider simplifying the model further or inspecting the data for issues.")

        # Return self to allow method chaining
        return self