"""ML model management class
For now, we use a dummy model, but the structure is ready
for a real scikit-learn or other ML model.
"""

import os

import joblib
import numpy as np


class MLModel:
    """Wrapper for our ML model"""

    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_version = "0.1.0"
        self.model_type = "DummyRegressor"
        self.expected_features = 4
        self.training_date = "2024-01-15"

    def load_model(self):
        """Load model from file or create a dummy model"""
        model_path = os.getenv("MODEL_PATH", "model.pkl")

        try:
            # Try to load existing model
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"✅ Model loaded from {model_path}")
            else:
                # Create dummy model for demo
                self._create_dummy_model()
                print("📝 Dummy model created for demo")

            self.is_loaded = True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to dummy model
            self._create_dummy_model()
            self.is_loaded = True

    def _create_dummy_model(self):
        """Create a dummy model for demonstration
        In reality, you would replace this with loading your real model
        """
        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression

        # Generate dummy data
        X, y = make_regression(
            n_samples=100, n_features=self.expected_features, noise=0.1, random_state=42
        )

        # Train simple model
        self.model = LinearRegression()
        self.model.fit(X, y)

        # Save for next time (optional)
        try:
            joblib.dump(self.model, "model.pkl")
        except Exception:
            pass  # Not critical if we can't save

    def predict(self, features: list[float]) -> float:
        """Make a prediction"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        # Validate features
        if len(features) != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features, got {len(features)}"
            )

        # Convert to numpy array and reshape for sklearn
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(features_array)[0]

        return float(prediction)

    def get_prediction_confidence(self, features: list[float]) -> float:
        """Calculate confidence score for the prediction
        This is dummy logic, but you could implement real confidence metrics
        """
        if not self.is_loaded:
            return 0.0

        # Dummy confidence based on feature variance
        # In reality, you would use methods like:
        # - Prediction intervals for regression
        # - Probability scores for classification
        # - Uncertainty quantification

        variance = np.var(features)
        confidence = min(0.95, max(0.5, 1.0 - variance / 10.0))

        return float(confidence)

    def is_ready(self) -> bool:
        """Check if model is ready"""
        return self.is_loaded and self.model is not None

    def get_version(self) -> str:
        """Return model version"""
        return self.model_version

    def get_model_type(self) -> str:
        """Return model type"""
        if self.model is not None:
            return type(self.model).__name__
        return self.model_type

    def get_expected_features(self) -> int:
        """Return number of expected features"""
        return self.expected_features

    def get_training_date(self) -> str:
        """Return training date"""
        return self.training_date
