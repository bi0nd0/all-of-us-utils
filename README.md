# aou_utils Library

## Overview

`aou_utils` is a custom Python library that provides various utilities and tools for data manipulation and query building. This library includes modules for handling datasets, constructing queries, calculating ages, generating cohorts, and creating concept tables.

## Installation

To install the `aou_utils` library, follow these steps:

1. **Clone the Repository**

   First, clone the repository from GitHub to your local machine:

   ```sh
   git clone https://github.com/username/all-of-us-utils.git
   ```

   Replace `https://github.com/username/all-of-us-utils.git` with the actual URL of your GitHub repository.

2. **Navigate to the Repository Directory**

   Change the directory to where the `setup.py` file is located:

   ```sh
   cd all-of-us-utils
   ```

3. **Install the Library in Editable Mode**

   Install the library using `pip` in editable mode. This allows you to see changes immediately without reinstalling:

   ```sh
   pip install -e .
   ```

## Usage

Once the library is installed, you can use it in your Python scripts or Jupyter notebooks. Below are examples of how to import and use the various modules and classes provided by the `aou_utils` library.

### Importing Modules

You can import all modules at once or import specific modules as needed.

#### Importing All Modules

To import all modules at once:

```python
from aou_utils import *
```

#### Importing Specific Modules

To import specific modules:

```python
from aou_utils import Utils
from aou_utils import QueryBuilder
from aou_utils import AgeCalculator
from aou_utils import CohortGenerator
from aou_utils import ConceptsTableMaker
```

### Example Usage

Below is an example of how to use the `QueryBuilder` and `Utils` modules:

```python
# Import necessary modules
from aou_utils import Utils
from aou_utils.QueryBuilder import QueryBuilder

# Set your dataset name
dataset = 'your_dataset_name'  # Replace with your actual dataset name

# Initialize the QueryBuilder
qb = QueryBuilder(dataset)

# Build a query
query = qb.selectDemography().include_concept_ids([4066824]).build()

# Execute the query and get the results as a DataFrame
person_df = Utils.get_dataframe(query)

# Display the first 5 rows of the DataFrame
person_df.head(5)

# Build a list of person IDs to use in other queries
person_id_list = person_df['person_id'].tolist()
```

## Modules

### Utils

The `Utils` module provides utility functions for data manipulation and query execution.

### QueryBuilder

The `QueryBuilder` module allows you to construct complex SQL queries based on various criteria.

### AgeCalculator

The `AgeCalculator` module provides functions for calculating ages based on birth dates.

### CohortGenerator

The `CohortGenerator` module helps in generating cohorts from datasets.

### ConceptsTableMaker

The `ConceptsTableMaker` module assists in creating concept tables from datasets.

## Contributing

If you would like to contribute to the development of `aou_utils`, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your forked repository.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
