"""
# Model Registry
# --------------
# A simple decorator-based registry that lets you plug in any
# sklearn-compatible model by name. Each registered builder function
# returns a (model_instance, param_grid) tuple so the generic trainer
# can run GridSearchCV on any model without knowing its internals.
#
# To add a new model:
#   1. Write a builder function below
#   2. Decorate it with @register("your_model_name")
#   3. Set model.active in config.yaml to "your_model_name"
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

REGISTRY = {}


def register(name):
    """
    Decorator that registers a model builder function under `name`.
    The builder must return (model_instance, param_grid_dict).
    """
    def decorator(fn):
        REGISTRY[name] = fn
        return fn
    return decorator


def get_model(name):
    """
    Look up a registered model builder by name and call it.

    Returns
    -------
    tuple : (sklearn estimator, dict)
        The unfitted model instance and its hyperparameter grid
        for GridSearchCV.

    Raises
    ------
    KeyError
        If the model name is not registered.
    """
    if name not in REGISTRY:
        available = ', '.join(REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return REGISTRY[name]()


# -------------------------------------------------------
# Registered Models
# -------------------------------------------------------

@register("logistic_regression")
def build_logistic_regression():
    """
    Logistic Regression with Elastic Net regularization.

    Why Elastic Net?
    - Combines L1 (Lasso) and L2 (Ridge) penalties
    - L1 drives uninformative feature coefficients to zero (feature selection)
    - L2 stabilizes correlated features
    - l1_ratio controls the mix: 0 = pure Ridge, 1 = pure Lasso

    Why SAGA solver?
    - Required for elastic net penalty
    - Efficient on large datasets
    """

    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=1000,
        random_state=42
    )

    # Grid of hyperparameters to search over:
    # C = inverse regularization strength (smaller C = more regularization)
    # l1_ratio = balance between L1 and L2 penalty
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    return model, param_grid


@register("logistic_regression_fast")
def build_logistic_regression_fast():
    """
    Logistic Regression without grid search.
    """
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.1,
        C=0.01,
        max_iter=1000,
        random_state=42
    )
    param_grid = {}
    return model, param_grid


@register("random_forest")
def build_random_forest():
    """
    Random Forest Classifier.

    Why Random Forest?
    - Handles non-linear relationships that logistic regression misses
    - Built-in feature importance via Gini impurity
    - Robust to outliers and doesn't require feature scaling
    - Good baseline for comparison against linear models
    """

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1  # use all CPU cores for parallel tree building
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    return model, param_grid
