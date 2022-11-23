import os

from catboost import CatBoostClassifier

from utils.paths import ENTITY_LINKER
from entity_linker.entity_linking_pipeline.candidates_ranger import ModelRanger


class CatboostRanger(ModelRanger):

    def _get_model_probas(self, x_test, ids):
        self._ml_model_path = os.path.join(ENTITY_LINKER, 'additional_data/model_cosine')
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(self._ml_model_path)
        model_result = loaded_model.predict_proba(x_test)[:, 1]
        result = list()
        for i, res in enumerate(model_result):
            result.append((ids[i], res))
        return result
