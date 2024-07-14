from .QueryBuilder import QueryBuilder
from .Utils import Utils
import pandas as pd

class ConceptsTableMaker:
    def __init__(self):
        pass
    
    def combine_concepts_counts(self, person_ids, concepts_group):
        data = []
        total_persons = len(person_ids)
        qb = QueryBuilder()

        for entry in concepts_group:
            label = entry['label']
            concept_ids = entry['ids']
            query = qb.include_concept_ids(concept_ids).include_persons(person_ids).build()
            df = Utils.get_dataframe(query)
            total = len(df)  # Get the count of the results
            percentage = (total / total_persons) * 100 if total_persons > 0 else 0
            data.append({'label': label, 'total': total, 'percentage': percentage})

        result_df = pd.DataFrame(data)
        return result_df