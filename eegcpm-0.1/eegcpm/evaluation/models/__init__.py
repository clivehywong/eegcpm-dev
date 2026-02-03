"""Model factory for different prediction models."""

from typing import Any, Dict, List, Optional, Union

import numpy as np


class ModelFactory:
    """Factory for creating prediction models."""

    @staticmethod
    def create(
        model_type: str,
        **kwargs,
    ) -> Any:
        """
        Create a model instance.

        Args:
            model_type: Type of model
            **kwargs: Model parameters

        Returns:
            Scikit-learn compatible model
        """
        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=kwargs.get("alpha", 1.0))

        elif model_type == "lasso":
            from sklearn.linear_model import Lasso
            return Lasso(alpha=kwargs.get("alpha", 1.0))

        elif model_type == "elasticnet":
            from sklearn.linear_model import ElasticNet
            return ElasticNet(
                alpha=kwargs.get("alpha", 1.0),
                l1_ratio=kwargs.get("l1_ratio", 0.5),
            )

        elif model_type == "svr":
            from sklearn.svm import SVR
            return SVR(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
            )

        elif model_type == "svm":
            from sklearn.svm import SVC
            return SVC(
                kernel=kwargs.get("kernel", "linear"),
                C=kwargs.get("C", 1.0),
                probability=True,
            )

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                random_state=42,
            )

        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                random_state=42,
            )

        elif model_type == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

        elif model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=kwargs.get("C", 1.0),
                max_iter=1000,
            )

        elif model_type == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            return LinearDiscriminantAnalysis()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_pipeline(
        model_type: str,
        scale: bool = True,
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None,
        **model_kwargs,
    ) -> Any:
        """
        Create a full prediction pipeline.

        Args:
            model_type: Type of model
            scale: Whether to standardize features
            feature_selection: Feature selection method (None, "univariate", "rfe")
            n_features: Number of features to select
            **model_kwargs: Model parameters

        Returns:
            Scikit-learn Pipeline
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps = []

        # Scaling
        if scale:
            steps.append(("scaler", StandardScaler()))

        # Feature selection
        if feature_selection == "univariate":
            from sklearn.feature_selection import SelectKBest, f_regression
            k = n_features or 100
            steps.append(("feature_selection", SelectKBest(f_regression, k=k)))

        elif feature_selection == "rfe":
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import Ridge
            k = n_features or 100
            steps.append(("feature_selection", RFE(Ridge(), n_features_to_select=k)))

        # Model
        model = ModelFactory.create(model_type, **model_kwargs)
        steps.append(("model", model))

        return Pipeline(steps)

    @staticmethod
    def grid_search(
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        cv: int = 5,
        scoring: str = "r2",
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.

        Args:
            model_type: Type of model
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid (uses defaults if None)
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            Best parameters and results
        """
        from sklearn.model_selection import GridSearchCV

        # Default parameter grids
        default_grids = {
            "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
            "svr": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
            "svm": {"C": [0.1, 1.0, 10.0]},
        }

        if param_grid is None:
            param_grid = default_grids.get(model_type, {})

        pipeline = ModelFactory.create_pipeline(model_type)

        # Adjust param names for pipeline
        pipeline_params = {f"model__{k}": v for k, v in param_grid.items()}

        grid_search = GridSearchCV(
            pipeline,
            pipeline_params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid_search.fit(X, y)

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
            "best_estimator": grid_search.best_estimator_,
        }
