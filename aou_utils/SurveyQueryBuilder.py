from .Utils import Utils

class SurveyQueryBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.selectQuery = None
        self.inclusion_criteria = []
        self.exclusion_criteria = []
    
    def get_concept_query(self, concept_ids):
        concept_ids_str = Utils.list_to_string(concept_ids, quote=False)
        return f"""SELECT criteria.person_id 
            FROM (
                SELECT DISTINCT person_id, entry_date, concept_id 
                FROM `{self.dataset}.cb_search_all_events` 
                WHERE concept_id IN (
                    SELECT DISTINCT c.concept_id 
                    FROM `{self.dataset}.cb_criteria` c 
                    JOIN (
                        SELECT CAST(cr.id as string) AS id
                        FROM `{self.dataset}.cb_criteria` cr
                        WHERE concept_id IN ({concept_ids_str})
                        AND full_text LIKE '%_rank1]%'
                    ) a ON (
                        c.path LIKE CONCAT('%.', a.id, '.%') 
                        OR c.path LIKE CONCAT('%.', a.id) 
                        OR c.path LIKE CONCAT(a.id, '.%') 
                        OR c.path = a.id
                    ) 
                    WHERE is_standard = 0 
                    AND is_selectable = 1
                ) 
                AND is_standard = 0
            ) criteria"""
    
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
    
    def reset(self):
        self.inclusion_criteria = []
        self.exclusion_criteria = []
        return self
    
    def select(self):
        def getQuery(condition):
            return f"""
                SELECT
                    answer.person_id,
                    answer.survey_datetime,
                    answer.survey,
                    answer.question_concept_id,
                    answer.question,
                    answer.answer_concept_id,
                    answer.answer,
                    answer.survey_version_concept_id,
                    answer.survey_version_name  
                FROM
                    `{self.dataset}.ds_survey` answer    
                WHERE
                    answer.PERSON_ID IN (
                        SELECT DISTINCT person_id  
                        FROM `{self.dataset}.cb_search_person` cb_search_person  
                        WHERE {condition}
                    )"""
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