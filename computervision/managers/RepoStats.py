#!/usr/bin/env python
# coding: utf-8
import json
import os
import pandas as pd
import sqlalchemy as sqlal

from collections import defaultdict
from constants import ENVS
from types import SimpleNamespace
from datetime import date

from managers.Repository import DataRepository
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
        recipe:SimpleNamespace, 
        rundate:date, 
        step:str='SKILL',
    ) -> dict:
        # TODO : check if trainStart works and fetches runs
        qry = sqlal.text(
            f"""SELECT * FROM TrainResuls tr
                JOIN TrainResultsEpoch tre ON tr.bestEpoch = tre.epoch AND tre.trainResultId = tr.id
                WHERE tr.step = :step
                AND tr.recipeCode = :recipeCode
                AND tr.trainStart = :rundate
                """)

        with self._get_connection() as connection:
            params = {
                'step': step,
                'recipeCode': recipe.name,
                'rundate': rundate,
            }
            df = pd.read_sql(qry, con=connection, params=params)
            print(df.head())
            # TODO : update to possibly take revalidation result
            return None if len(df) == 0 else df.loc[0, 'validationResult']

    def get_epochs_no_improvement(
        self,
        recipe: SimpleNamespace, 
        rundate: date, 
        validation_results: dict,
        compare_result_method:str,
        step:str='SKILL',
    ) -> int:
        """
        Count how many epochs without improvement compared to current validation results.
        
        Note: Skips the most recent epoch (just saved) and compares against previous epochs.
        
        Returns:
            int: Number of consecutive epochs without improvement
        """
        qry = sqlal.text(
            """SELECT tre.epoch, tre.validationResults 
               FROM TrainResults tr
               JOIN TrainResultsEpoch tre ON tr.id = tre.trainResultId
               WHERE tr.step = :step
               AND tr.recipeCode = :recipeCode
               AND tr.trainStart = :rundate
               ORDER BY tre.epoch DESC
            """)

        with self._get_connection() as connection:
            params = {
                'step': step,
                'recipeCode': recipe.name,
                'rundate': rundate,
            }
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

    def check_is_best_model(
        self,
        recipe: SimpleNamespace, 
        rundate: date, 
        validation_results: dict,
        compare_result_method:str,
        step:str='SKILL',
    ) -> dict[str, bool]:
        """Returns is_best_of = {
            'all' : bool,
            'recipe' : bool,
            'architecture' : bool
        }
        Updates the isBestOf* flags in the database accordingly.
        """
        model_is_best_of = {}
        
        with self._get_connection() as connection:
            # Get current TrainResult ID for this run
            current_train_result = connection.execute(
                sqlal.text("""
                    SELECT id FROM TrainResults 
                    WHERE step = :step 
                    AND recipeCode = :recipeCode 
                    AND trainStart = :trainStart
                """),
                {'step': step, 'recipeCode': recipe.name, 'trainStart': rundate}
            ).fetchone()
            
            current_train_result_id = current_train_result[0] if current_train_result else None
            
            # For 'all': compare against current best of all models
            qry_all = sqlal.text("""
                SELECT tre.validationResults, tr.id FROM TrainResults tr
                JOIN TrainResultsEpoch tre ON tr.id = tre.trainResultId AND tr.bestEpoch = tre.epoch
                WHERE tr.step = :step
                AND tr.isBestOfAll = True
                ORDER BY tr.trainStart DESC
                LIMIT 1
            """)
            
            # For 'recipe': compare against current best of this recipe
            qry_recipe = sqlal.text("""
                SELECT tre.validationResults, tr.id FROM TrainResults tr
                JOIN TrainResultsEpoch tre ON tr.id = tre.trainResultId AND tr.bestEpoch = tre.epoch
                WHERE tr.step = :step
                AND tr.recipeCode = :recipeCode
                AND tr.isBestOfRecipe = True
                ORDER BY tr.trainStart DESC
                LIMIT 1
            """)
            
            # For 'architecture': compare against current best of this architecture
            qry_arch = sqlal.text("""
                SELECT tre.validationResults, tr.id FROM TrainResults tr
                JOIN TrainResultsEpoch tre ON tr.id = tre.trainResultId AND tr.bestEpoch = tre.epoch
                WHERE tr.step = :step
                AND JSON_EXTRACT(tr.recipe, '$.model') = :model
                AND tr.isBestOfArchitecture = True
                ORDER BY tr.trainStart DESC
                LIMIT 1
            """)
            
            params_base = {'step': step}
            params_recipe = {**params_base, 'recipeCode': recipe.name}
            params_arch = {**params_base, 'model': recipe.model}
            
            # Get best results for each comparison type
            for best_of_type, qry, params, col_name in [
                ('all', qry_all, params_base, 'isBestOfAll'),
                ('recipe', qry_recipe, params_recipe, 'isBestOfRecipe'),
                ('architecture', qry_arch, params_arch, 'isBestOfArchitecture'),
            ]:
                result = connection.execute(qry, params).fetchone()
                
                if result is None:
                    # No previous best, so this is the best
                    model_is_best_of[best_of_type] = True
                    if current_train_result_id:
                        update_qry = sqlal.text(f"""
                            UPDATE TrainResults 
                            SET {col_name} = True
                            WHERE id = :train_result_id
                        """)
                        connection.execute(update_qry, {'train_result_id': current_train_result_id})
                        connection.commit()
                else:
                    other_validation_results = json.loads(result[0])
                    other_train_result_id = result[1]
                    
                    is_best = self.compare_validation_results(
                        validation_results, 
                        other_validation_results, 
                        compare_result_method
                    )
                    
                    if is_best and current_train_result_id:
                        # Update current model as best
                        update_current = sqlal.text(f"""
                            UPDATE TrainResults 
                            SET {col_name} = True
                            WHERE id = :train_result_id
                        """)
                        connection.execute(update_current, {'train_result_id': current_train_result_id})
                        
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

    def save_epoch_results(self, recipe: SimpleNamespace, rundate: date, epoch: int, validation_results: dict, step: str = 'SKILL'):
        """
        Save epoch results to TrainResults and TrainResultsEpoch tables.
        
        Parameters:
            recipe (SimpleNamespace): Recipe information with name, model, etc.
            rundate (date): Training run date
            epoch (int): Epoch number
            validation_results (dict): Validation metrics and results
            step (str): Step name (SKILL, LOCALIZE, etc.)
        """
        with self._get_connection() as connection:
            # Check if TrainResult exists for this run
            check_qry = sqlal.text("""
                SELECT id FROM TrainResults 
                WHERE step = :step 
                AND recipeCode = :recipeCode 
                AND trainStart = :trainStart
            """)
            
            params = {
                'step': step,
                'recipeCode': recipe.name,
                'trainStart': rundate,
            }
            
            result = connection.execute(check_qry, params).fetchone()
            train_result_id = None
            
            if result is None:
                # Insert new TrainResult record
                insert_result_qry = sqlal.text("""
                    INSERT INTO TrainResults 
                    (step, recipeCode, recipe, bestEpoch, revalidationResults, isBestOfAll, isBestOfRecipe, isBestOfArchitecture, trainStart, isRunning)
                    VALUES 
                    (:step, :recipeCode, :recipe, :bestEpoch, :revalidationResults, :isBestOfAll, :isBestOfRecipe, :isBestOfArchitecture, :trainStart, :isRunning)
                """)
                
                insert_params = {
                    'step': step,
                    'recipeCode': recipe.name,
                    'recipe': json.dumps(vars(recipe)),
                    'bestEpoch': epoch,
                    'revalidationResults': json.dumps({}),
                    'isBestOfAll': False,
                    'isBestOfRecipe': False,
                    'isBestOfArchitecture': False,
                    'trainStart': rundate,
                    'isRunning': True,
                }
                
                connection.execute(insert_result_qry, insert_params)
                connection.commit()
                
                # Get the newly created ID
                result = connection.execute(check_qry, params).fetchone()
            
            train_result_id = result[0]
            
            # Insert epoch results into TrainResultsEpoch
            insert_epoch_qry = sqlal.text("""
                INSERT INTO TrainResultsEpoch 
                (trainResultId, epoch, validationResults)
                VALUES 
                (:trainResultId, :epoch, :validationResults)
            """)
            
            epoch_params = {
                'trainResultId': train_result_id,
                'epoch': epoch,
                'validationResults': json.dumps(validation_results),
            }
            
            connection.execute(insert_epoch_qry, epoch_params)
            connection.commit()

    def compare_validation_results(self, current_results: dict, other_results: dict, method: str):
        assert method in VALIDATION_COMPARE_METHODS or method is None, f"❌ Unknown compare method ({method})"
        if method is None:
            print(f"⚠️ Compare method is None")
            method = 'f1_avg'
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
        current_f1_avg = current_results.get('metric_avg_of_props')['f1']
        weighted_other_f1_avg = other_results.get('metric_avg_of_props')['f1'] * quadratic_weight
        return current_f1_avg > weighted_other_f1_avg

    def simple_f1(self, current_results: dict, other_results: dict) -> bool:
        return current_results.get('metric_avg_of_props')['f1'] > other_results.get('metric_avg_of_props')['f1']

REPO_STATS = RepoStats()

