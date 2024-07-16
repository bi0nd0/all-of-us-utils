from aou_utils import Utils
from aou_utils.SurveyQueryBuilder import SurveyQueryBuilder

# Set your dataset name
dataset = 'your_dataset_name'  # Replace with your actual dataset name

# Initialize the QueryBuilder
qb = SurveyQueryBuilder(dataset)

# Build a query
query = qb.select().include_concept_ids([1585873, 1585860]).build()

# Execute the query and get the results as a DataFrame
df = Utils.get_dataframe(query)

# Display the first 5 rows of the DataFrame
print(person_df.head(5))

# Build a list of person IDs to use in other queries
person_id_list = df['person_id'].tolist()
print(person_id_list)