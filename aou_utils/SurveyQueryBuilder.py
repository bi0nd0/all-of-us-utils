from .Utils import Utils

class SurveyQueryBuilder:
    def __init__(self, dataset):
        self.dataset = dataset
        self.selectQuery = None
        self.inclusion_criteria = []
        self.exclusion_criteria = []

    def get_concept_query(self, concept_ids):
        concept_ids_str = Utils.list_to_string(concept_ids, quote=False)
        return f"{concept_ids_str}"

    def include_concept_ids(self, concept_ids):
        concept_query = self.get_concept_query(concept_ids)
        query = f"question_concept_id IN ({concept_query})"
        self.inclusion_criteria.append(query)
        return self

    def exclude_concept_ids(self, concept_ids):
        concept_query = self.get_concept_query(concept_ids)
        query = f"question_concept_id NOT IN ({concept_query})"
        self.exclusion_criteria.append(query)
        return self

    def include_persons(self, person_ids):
        ids_str = Utils.list_to_string(person_ids, quote=False)
        query = f"person_id IN ({ids_str})"
        self.inclusion_criteria.append(query)
        return self

    def exclude_persons(self, person_ids):
        ids_str = Utils.list_to_string(person_ids, quote=False)
        query = f"person_id NOT IN ({ids_str})"
        self.exclusion_criteria.append(query)
        return self

    def reset(self):
        self.inclusion_criteria = []
        self.exclusion_criteria = []
        return self

    def select(self):
        def getQuery(condition):
            return f"""
                WITH persons AS (
                    SELECT
                        person_id,
                        birth_datetime,
                        concept_name AS sex_at_birth
                    FROM
                        `{self.dataset}.person`
                    LEFT JOIN `{self.dataset}.concept` ON concept_id = sex_at_birth_concept_id
                ),
                surveys AS (
                    SELECT
                        person_id,
                        survey_datetime,
                        survey,
                        question_concept_id,
                        question,
                        answer_concept_id,
                        answer,
                        survey_version_concept_id,
                        survey_version_name,
                        ROW_NUMBER() OVER (PARTITION BY person_id
                                           ORDER BY survey_datetime DESC) AS recency_rank
                    FROM
                        `{self.dataset}.ds_survey`
                    WHERE
                        {condition}
                )
                SELECT
                    persons.person_id,
                    persons.birth_datetime,
                    persons.sex_at_birth,
                    surveys.survey_datetime,
                    surveys.survey,
                    surveys.question_concept_id,
                    surveys.question,
                    surveys.answer_concept_id,
                    surveys.answer,
                    surveys.survey_version_concept_id,
                    surveys.survey_version_name
                FROM
                    surveys
                INNER JOIN
                    persons ON persons.person_id = surveys.person_id
                WHERE
                    surveys.recency_rank = 1
                ORDER BY
                    persons.person_id, surveys.survey_datetime;

            """
        self.selectQuery = getQuery
        return self

    def build(self, reset=True):
        # Combine inclusion and exclusion criteria
        all_criteria = self.inclusion_criteria + self.exclusion_criteria
        # Join all criteria with ' AND '
        condition = ' AND '.join(all_criteria) if all_criteria else '1=1'
        query = self.selectQuery(condition)
        if reset:
            self.reset()
        return query
