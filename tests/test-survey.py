from aou_utils import Utils
from aou_utils.SurveyQueryBuilder import SurveyQueryBuilder

# Set your dataset name
dataset = 'your_dataset_name'  # Replace with your actual dataset name

# Initialize the QueryBuilder
qb = SurveyQueryBuilder(dataset)

# Build a query
query = qb.select().include_concept_ids([1585873, 1585860, 1234]).exclude_persons([1,2,3,4,5,6,7,8,9,0]).build()

# Display the first 5 rows of the DataFrame
print(query)