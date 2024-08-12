from .QueryBuilder import QueryBuilder
from .Utils import Utils
import pandas as pd

class ConceptsTableMaker:
    def __init__(self):
        pass
    
    def combine_concepts_counts(self, dataset, person_ids, concepts_group):
        """
        Combines the counts of a single comorbidity into a DataFrame.
        
        Parameters:
        - dataset: The dataset to be used for querying.
        - person_ids: The list of person IDs to be considered.
        - concepts_group: A list containing one dictionary with 'label', 'ids', and 'code' for a comorbidity.
        
        Returns:
        - A DataFrame with the total count and percentage of persons with the given comorbidity.
        """
        data = []
        total_persons = len(person_ids)
        qb = QueryBuilder(dataset)
        qb = qb.selectDemography()
        
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

    def combine_all_comorbidities_counts(self, dataset, person_ids, comorbidities):
        """
        Combines the counts of multiple comorbidities into a single DataFrame.

        Parameters:
        - dataset: The dataset to be used for querying.
        - person_ids: The list of person IDs to be considered.
        - comorbidities: A list of dictionaries where each dictionary contains 'label', 'ids', and 'code' for a comorbidity.

        Returns:
        - A DataFrame containing the combined counts of all comorbidities.
        """
        combined_df = pd.DataFrame()

        for comorbidity in comorbidities:
            # Apply the combine_concepts_counts function for each comorbidity
            result_df = self.combine_concepts_counts(dataset, person_ids, [comorbidity])
            
            # Add a column for the comorbidity label
            result_df['comorbidity'] = comorbidity['label']
            
            # Concatenate the result to the combined DataFrame
            combined_df = pd.concat([combined_df, result_df], ignore_index=True)

        return combined_df
