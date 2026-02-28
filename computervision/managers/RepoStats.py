#!/usr/bin/env python
# coding: utf-8
import json
import os
import pandas as pd
import sqlalchemy as sqlal

from collections import defaultdict
from constants import ENVS
from types import SimpleNamespace
from typing import TypedDict, Any
from datetime import date, datetime
from domain.types import IsBestOfDict

from managers.Repository import DataRepository
from uuid import UUID

BEST_OF_TYPES = [ 'all', 'recipe', 'architecture' ]
VALIDATION_COMPARE_METHODS = [
    'quadratic_validation_length_weighted_f1',
    'f1_avg',
]

class RepoStats(DataRepository):
    VideoNames = {} # pandas dataframe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_validation_result(
        self,
        recipe: SimpleNamespace,
        runtime: datetime,
        step: str='SKILL',
    ) -> dict:
        # TODO : check if createdAt works and fetches runs
        qry = sqlal.text(
            f"""SELECT * FROM TrainResuls tr
                JOIN TrainResultsEpoch tre ON tr.bestEpoch = tre.epoch AND tre.trainResultId = tr.id
                WHERE tr.step = :step
                AND tr.recipeCode = :recipeCode
                AND tr.createdAt = :runtime
                """)

        with self._get_connection() as connection:
            params = {
                'step': step,
                'recipeCode': recipe.name,
                'runtime': runtime,
            }
            df = pd.read_sql(qry, con=connection, params=params)
            # TODO : update to possibly take revalidation result
            return None if len(df) == 0 else df.loc[0, 'validationResult']

    def get_epochs_no_improvement(
        self,
        train_result_id: UUID,
        validation_results: dict,
        compare_result_method:str,
    ) -> int:
        """
        Count how many epochs without improvement compared to current validation results.

        ⚠️ Note: Make sure the epoch results are already saved to DB.
        This methods goes to DB to check and iterate

        Parameters:
            train_result_id (int): Id of the train_result.
            validation_results (dict): Current validation results
            compare_result_method (str): Method name to compare

        Returns:
            int: Number of consecutive epochs without improvement
        """
        qry = sqlal.text(
            """SELECT tre.epoch, tre.validationResults
               FROM TrainResults tr
               JOIN TrainResultsEpoch tre ON tr.id = tre.trainResultId
               WHERE tr.id = :train_result_id
               ORDER BY tre.epoch DESC
            """)

        with self._get_connection() as connection:
            params = { 'train_result_id': train_result_id }
            df = pd.read_sql(qry, con=connection, params=params)

            # Need at least 2 epochs to compare (current + previous)
            if len(df) < 2:
                return 0

            # Parse validationResults from JSON
            df['validationResults'] = df['validationResults'].apply(json.loads)

            # Skip the first row (most recently saved epoch) and count epochs without improvement
            epochs_no_improvement = 0
            for idx, row in df.iloc[1:].iterrows():
                other_validation_results = row['validationResults']

                is_improvement = self.compare_validation_results(validation_results, other_validation_results, compare_result_method)

                if not is_improvement:
                    epochs_no_improvement += 1
                else:
                    break

            return epochs_no_improvement

    def get_train_result_which_is_best_of(self, best_of: str, recipe: SimpleNamespace, isTestrun: bool, step: str) -> tuple[Any, str]:
        f"""
        Docstring for get_train_result_which_is_best_of

        :param self: RepoStats instance
        :param best_of: value in [{','.join(BEST_OF_TYPES)}]
        :type best_of: str
        :param recipe: Recipe information with name, model, etc.
        :type recipe: SimpleNamespace
        :param isTestrun: Indicates wheter this train round is for testing or not
        :type isTestrun: bool
        :param step: Step name (SKILL, LOCALIZE, etc.)
        :type step: str

        :return: result = connection.execute && best_of column name (e.g. isBestOfAll, isBestOfRecipe)
        :rtype: tuple[Any, str]
        """
        assert best_of in BEST_OF_TYPES, f"❌ Invalid best of type ({best_of}) fetching best train_result"
        # TODO :
        qry_filter = ''
        qry_params = {
            'step': step,
            'isTestrun': isTestrun
        }
        col_name = None
        match best_of:
            case 'all':
                col_name = 'isBestOfAll'
                qry_filter = f"""tr.isBestOfAll = 1"""
            case 'recipe':
                col_name = 'isBestOfRecipe'
                qry_filter = f"""
                tr.recipeCode = :recipeCode
                AND tr.isBestOfRecipe = 1
                """
                qry_params = { **qry_params, 'recipeCode': recipe.name }
            case 'architecture':
                col_name = 'isBestOfArchitecture'
                qry_filter = f"""
                JSON_EXTRACT(tr.recipe, '$.architecture') = :recipeArchitecture
                AND tr.isBestOfArchitecture = True
                """
                qry_params = { **qry_params, 'recipeArchitecture' : recipe.architecture }

        qry_is_best_of = sqlal.text(f"""
            SELECT tre.validationResults, tr.id FROM TrainResults tr
            JOIN TrainResultsEpoch tre ON tr.id = tre.trainResultId AND tr.bestEpoch = tre.epoch
            WHERE tr.step = :step
            AND {qry_filter}
            AND tr.isTestrun = :isTestrun
            ORDER BY tr.createdAt DESC
            LIMIT 1
        """)
        with self._get_connection() as connection:
            result = connection.execute(qry_is_best_of, qry_params).mappings().fetchone()

        return result, col_name

    def check_is_best_model(
        self,
        train_result_id: UUID,
        recipe: SimpleNamespace,
        validation_results: dict,
        compare_result_method: str,
        isTestrun: bool,
        step: str,
    ) -> IsBestOfDict:
        """
        Determines whether this model is best-of across different dimensions
        and updates the isBestOf* flags in the database.

        Parameters:
            train_result_id (int): Id of the train_result.
            recipe (SimpleNamespace): Recipe information with name, model, etc.
            validation_results (dict): Validation metrics and results
            compare_result_method: (str): Method to use for comparing which model is best
            testrun (bool): Indicates wheter this train round is for testing or not
            step (str): Step name (SKILL, LOCALIZE, etc.)

        :return: best of model types
        :rtype: IsBestOfDict

        """
        assert isinstance(train_result_id, int), f"❌ Train result id is not an int, got {type(train_result_id)}"
        assert self.exists_train_result(train_result_id), f"❌ Train result with id: {train_result_id} does not exist"

        model_is_best_of = IsBestOfDict()

        with self._get_connection() as connection:
            # Get best results for each comparison type
            for best_of_type in BEST_OF_TYPES:
                result, col_name = self.get_train_result_which_is_best_of(
                    best_of=best_of_type,
                    recipe=recipe,
                    isTestrun=isTestrun,
                    step=step
                )

                if result is None:
                    f"⚠️ Previous model best of {best_of_type} of is None"
                    # No previous best, so this is the best
                    model_is_best_of[best_of_type] = True

                    update_qry = sqlal.text(f"""
                        UPDATE TrainResults
                        SET {col_name} = True
                        WHERE id = :train_result_id
                    """)
                    connection.execute(update_qry, {'train_result_id': train_result_id})
                    connection.commit()
                else:
                    # TODO : use revalidation results later if added.
                    other_validation_results = json.loads(result['validationResults'])
                    other_train_result_id = result['id']

                    is_best = self.compare_validation_results(
                        validation_results,
                        other_validation_results,
                        compare_result_method
                    )

                    if is_best:
                        # Update current model as best
                        update_current = sqlal.text(f"""
                            UPDATE TrainResults
                            SET {col_name} = True
                            WHERE id = :train_result_id
                        """)
                        connection.execute(update_current, {'train_result_id': train_result_id})

                        # Set old best to False
                        update_old = sqlal.text(f"""
                            UPDATE TrainResults
                            SET {col_name} = False
                            WHERE id = :train_result_id
                        """)
                        connection.execute(update_old, {'train_result_id': other_train_result_id})
                        connection.commit()

                    model_is_best_of[best_of_type] = is_best

        return model_is_best_of

    def add_train_result(self, recipe: SimpleNamespace, testrun: bool, step: str = 'SKILL') -> int:
        """
        Add a train result.

        Parameters:
            recipe (SimpleNamespace): Recipe information with name, model, etc.
            testrun (bool): Indicates wheter this train round is for testing or not
            step (str): Step name (SKILL, LOCALIZE, etc.)

        Returns:
            train_result_id (int)
        """
        with self._get_connection() as connection:
            # Insert new TrainResult record
            insert_result_qry = sqlal.text("""
                INSERT INTO TrainResults
                (step, recipeCode, recipe, bestEpoch, revalidationResults, isBestOfAll, isBestOfRecipe, isBestOfArchitecture, createdAt, isTestrun, createdAt)
                VALUES
                (:step, :recipeCode, :recipe, :bestEpoch, :revalidationResults, :isBestOfAll, :isBestOfRecipe, :isBestOfArchitecture, :createdAt, :isTestrun, :createdAt)
            """)

            insert_params = {
                'step': step,
                'recipeCode': recipe.name,
                'recipe': json.dumps(vars(recipe)),
                'bestEpoch': 1,
                'revalidationResults': json.dumps({}),
                'isBestOfAll': False,
                'isBestOfRecipe': False,
                'isBestOfArchitecture': False,
                'createdAt': datetime.now(),
                'createdAt': datetime.now(),
                'isTestrun': testrun,
            }

            result = connection.execute(insert_result_qry, insert_params)
            train_result_id: UUID = result.lastrowid
            connection.commit()

            return train_result_id

    def exists_train_result(self, train_result_id: UUID) -> bool:
        with self._get_connection() as connection:
            check_qry = sqlal.text("""
                SELECT id FROM TrainResults
                WHERE id = :id
            """)

            params = {
                'id': train_result_id,
            }

            result = connection.execute(check_qry, params).fetchone()

        return result is not None

    def update_train_result(self, train_result_id: UUID, updated_params: dict):
        """
        Update a train result.

        Parameters:
            train_result_id (int): Id of the train_result.
            updated_params (dict): Parameters to update
        """
        TRAIN_RESULTS_TABLE_NAME = 'TrainResults'
        columns = self.get_table_columns(TRAIN_RESULTS_TABLE_NAME)
        column_names = {col["name"] for col in columns}
        assert isinstance(train_result_id, int), f"❌ train_result_id is not an int ({train_result_id}, {type(train_result_id)})"
        assert isinstance(updated_params, dict), f"❌ updated_params is not a dict ({type(updated_params)})"

        with self._get_connection() as connection:
            # Check if TrainResult exists for this ID
            if not self.exists_train_result(train_result_id):
                raise ValueError(f"❌ train_result_id ({train_result_id}) does not exist")

            # Check if all updated parameters are columns:
            for param in updated_params:
                if param not in column_names:
                    raise ValueError(f"❌ '{param}' is not a column in {TRAIN_RESULTS_TABLE_NAME}")

            # Insert new TrainResult record
            update_values = ', '.join(
                f'{col_name} = :{col_name}' for col_name in updated_params.keys()
            )
            update_result_qry = sqlal.text(f"""
                UPDATE {TRAIN_RESULTS_TABLE_NAME}
                SET {update_values}
                WHERE id = :id
            """)

            connection.execute(update_result_qry, {**updated_params, 'id': train_result_id})
            connection.commit()

    def save_epoch_results(self, train_result_id: UUID, epoch: int, validation_results: dict):
        """
        Save epoch results to TrainResults and TrainResultsEpoch tables.

        Parameters:
            train_result_id (int): Id of the train_result.
            epoch (int): Epoch number
            validation_results (dict): Validation metrics and results
        """
        with self._get_connection() as connection:
            # Check if TrainResult exists for this run
            check_qry = sqlal.text("""
                SELECT id FROM TrainResults
                WHERE id = :id
            """)

            params = { 'id': train_result_id }
            result = connection.execute(check_qry, params).fetchone()
            assert result is not None, f"❌ Train result id does not exist ({train_result_id})"

            # Insert epoch results into TrainResultsEpoch
            insert_epoch_qry = sqlal.text("""
                INSERT INTO TrainResultsEpoch
                (trainResultId, epoch, validationResults, createdAt)
                VALUES
                (:trainResultId, :epoch, :validationResults, :createdAt)
            """)

            epoch_params = {
                'trainResultId': train_result_id,
                'epoch': epoch,
                'validationResults': json.dumps(validation_results),
                'createdAt': datetime.now()
            }

            connection.execute(insert_epoch_qry, epoch_params)
            connection.commit()

    def compare_validation_results(self, current_results: dict, other_results: dict, method: str) -> bool:
        """
        Docstring for compare_validation_results

        :param self: RepoStats
        :param current_results: validation_results
        :type current_results: dict
        :param other_results: validation_results
        :type other_results: dict
        :param method: name of the method to use for comparison
        :type method: str
        :return: Returns whether current_results is better than other_results
        :rtype: bool
        """
        if method is None:
            print(f"⚠️ Compare method is None")
            method = 'f1_avg'
        assert current_results is not None, f"❌ current_results is None"
        assert other_results is not None, f"❌ other_results is None"
        assert method in VALIDATION_COMPARE_METHODS, f"❌ Unknown compare method ({method})"
        match method:
            case 'f1_avg':
                return self.simple_f1(current_results, other_results)
            case 'quadratic_validation_length_weighted_f1':
                return self.quadratic_validation_length_weighted_f1(current_results, other_results)

    ##########
    # Compare result methods
    ##########
    def quadratic_validation_length_weighted_f1(self, current_results: dict, other_results: dict) -> bool:
        """
        c = current size
        o = other size
        f1_current = current f1 avg
        f1_other = other f1 avg

        Returns: f1_current > f1_other * (o/c) ** 2
        """
        quadratic_weight = (other_results['val_length'] / current_results['val_length']) ** 2
        current_f1_avg = current_results.get('metric_avg_of_layers')['f1']
        if other_results is None or other_results.get('metric_avg_of_layers') is None:
            return True
        other_f1_avg = other_results.get('metric_avg_of_layers')['f1']
        weighted_other_f1_avg = other_f1_avg * quadratic_weight
        return current_f1_avg > weighted_other_f1_avg

    def simple_f1(self, current_results: dict, other_results: dict) -> bool:
        return current_results.get('metric_avg_of_layers')['f1'] > other_results.get('metric_avg_of_layers')['f1']

REPO_STATS = RepoStats()

