import sys
import os

# Set the PYTHONPATH to include the library directory
sys.path.append(os.path.abspath('../'))



from aou_utils import Utils
from aou_utils.SurveyQueryBuilder import SurveyQueryBuilder

# Set your dataset name
dataset = '{dataset}'  # Replace with your actual dataset name

# Initialize the QueryBuilder
qb = SurveyQueryBuilder(dataset)

# Build a query
query = qb.select().include_concept_ids([1585873, 1585860]).include_persons([1,2,3]).exclude_persons([1003548]).build()

# Display the first 5 rows of the DataFrame
print(query)