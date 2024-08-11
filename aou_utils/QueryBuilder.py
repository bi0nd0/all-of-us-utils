from .Utils import Utils
from datetime import datetime

class QueryBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.selectQuery = None
        self.inclusion_criteria = []
        self.exclusion_criteria = []
    
    def get_concept_query(self, concept_ids):
        concept_ids_str = Utils.list_to_string(concept_ids, quote=False)
        return f"""SELECT
                criteria.person_id 
            FROM
                (SELECT
                    DISTINCT person_id, entry_date, concept_id 
                FROM
                    `{self.dataset}.cb_search_all_events` 
                WHERE
                    (concept_id IN(SELECT
                        DISTINCT c.concept_id 
                    FROM
                        `{self.dataset}.cb_criteria` c 
                    JOIN
                        (SELECT
                            CAST(cr.id as string) AS id       
                        FROM
                            `{self.dataset}.cb_criteria` cr       
                        WHERE
                            concept_id IN ({concept_ids_str})       
                            AND full_text LIKE '%_rank1]%'      ) a 
                            ON (c.path LIKE CONCAT('%.', a.id, '.%') 
                            OR c.path LIKE CONCAT('%.', a.id) 
                            OR c.path LIKE CONCAT(a.id, '.%') 
                            OR c.path = a.id) 
                    WHERE
                        is_standard = 1 
                        AND is_selectable = 1) 
                    AND is_standard = 1 )) criteria"""
    
    def include_concept_ids(self, concept_ids):
        concept_query = self.get_concept_query(concept_ids)
        query = f"""cb_search_person.person_id IN ({concept_query})"""
        self.inclusion_criteria.append(query)
        return self
    
    def exclude_concept_ids(self, concept_ids):
        concept_query = self.get_concept_query(concept_ids)
        query = f"""cb_search_person.person_id NOT IN ({concept_query})"""
        self.exclusion_criteria.append(query)
        return self
    
    def include_persons(self, persons_ids):
        ids_str = Utils.list_to_string(persons_ids, quote=False)
        query = f"""cb_search_person.person_id IN ({ids_str})"""
        self.inclusion_criteria.append(query)
        return self
        
    def exclude_persons(self, persons_ids):
        ids_str = Utils.list_to_string(persons_ids, quote=False)
        query = f"""cb_search_person.person_id NOT IN ({ids_str})"""
        self.exclusion_criteria.append(query)
        return self
    
    def has_ehr_data(self, has_ehr=1):
        ehr_value = 1 if has_ehr else 0
        query = f"cb_search_person.has_ehr_data = {ehr_value}"
        self.inclusion_criteria.append(query)
        return self
    
    def calc_age(self, date, birth_date_key='birth_datetime'):
        # Converts the provided date string to a datetime object for consistency
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        date_str = date_obj.strftime('%Y-%m-%d')
        
        # Store the original selectQuery function
        original_select_query = self.selectQuery
        
        # Modify the selectQuery to include a calculated age column
        self.selectQuery = lambda condition: original_select_query(condition).replace(
            'FROM', f""", 
                    DATE_DIFF('{date_str}', person.{birth_date_key}, YEAR) AS age 
                    FROM""")
        return self

    
    def reset(self):
        self.inclusion_criteria = []
        self.exclusion_criteria = []
        return self
    
    def selectDemography(self):
        def getQuery(condition):
            return f"""
                SELECT
                    person.person_id,
                    person.gender_concept_id,
                    p_gender_concept.concept_name as gender,
                    person.birth_datetime as date_of_birth,
                    person.race_concept_id,
                    p_race_concept.concept_name as race,
                    person.ethnicity_concept_id,
                    p_ethnicity_concept.concept_name as ethnicity,
                    person.sex_at_birth_concept_id,
                    p_sex_at_birth_concept.concept_name as sex_at_birth 
                FROM
                    `{self.dataset}.person` person 
                LEFT JOIN
                    `{self.dataset}.concept` p_gender_concept 
                        ON person.gender_concept_id = p_gender_concept.concept_id 
                LEFT JOIN
                    `{self.dataset}.concept` p_race_concept 
                        ON person.race_concept_id = p_race_concept.concept_id 
                LEFT JOIN
                    `{self.dataset}.concept` p_ethnicity_concept 
                        ON person.ethnicity_concept_id = p_ethnicity_concept.concept_id 
                LEFT JOIN
                    `{self.dataset}.concept` p_sex_at_birth_concept 
                        ON person.sex_at_birth_concept_id = p_sex_at_birth_concept.concept_id  
                WHERE
                    person.PERSON_ID IN (SELECT
                        distinct person_id  
                    FROM
                        `{self.dataset}.cb_search_person` cb_search_person  
                    WHERE {condition}) """
        self.selectQuery = getQuery
        return self
        

    def build(self, reset=True):
        # Combine inclusion and exclusion criteria
        all_criteria = self.inclusion_criteria + self.exclusion_criteria
        # Join all criteria with ' AND '
        condition = ' AND '.join(all_criteria) if all_criteria else '1=1'
        query = self.selectQuery(condition)
        if(reset == True): self.reset()

        return query