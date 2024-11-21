from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class ModelConfig:
    name: str
    model_class: Any
    default_params: Dict[str, Any]
    param_ranges: Dict[str, Dict[str, Any]]

class ModelRegistry:
    def __init__(self):
        self.models = {
            'random_forest': ModelConfig(
                name='Random Forest',
                model_class=RandomForestClassifier,
                default_params={'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
                param_ranges={
                    'n_estimators': {'min': 10, 'max': 200, 'step': 10, 'type': 'int'},
                    'max_depth': {'min': 1, 'max': 20, 'step': 1, 'type': 'int'},
                    'min_samples_split': {'min': 2, 'max': 10, 'step': 1, 'type': 'int'}
                }
            ),
            'gradient_boosting': ModelConfig(
                name='Gradient Boosting',
                model_class=GradientBoostingClassifier,
                default_params={'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
                param_ranges={
                    'n_estimators': {'min': 10, 'max': 200, 'step': 10, 'type': 'int'},
                    'learning_rate': {'min': 0.01, 'max': 1.0, 'step': 0.01, 'type': 'float'},
                    'max_depth': {'min': 1, 'max': 10, 'step': 1, 'type': 'int'}
                }
            ),
            'svm': ModelConfig(
                name='SVM',
                model_class=SVC,
                default_params={'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                param_ranges={
                    'C': {'min': 0.1, 'max': 10.0, 'step': 0.1, 'type': 'float'},
                    'kernel': {'options': ['rbf', 'linear', 'poly'], 'type': 'categorical'},
                    'gamma': {'options': ['scale', 'auto'], 'type': 'categorical'}
                }
            )
        }
    
    def get_model(self, model_name: str, params: Dict[str, Any] = None) -> Any:
        if params is None:
            params = self.models[model_name].default_params
        return self.models[model_name].model_class(**params)