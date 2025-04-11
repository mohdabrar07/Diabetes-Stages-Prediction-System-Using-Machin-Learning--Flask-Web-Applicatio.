import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

class DiabetesRiskPredictor:
    def __init__(self, model_path=None):
        """Initialize the diabetes risk predictor with optional pre-trained model."""
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.is_fitted = False

    def validate_input(self, data):
        """Validate input data ranges and types."""
        validation_rules = {
            'Pregnancies': {'min': 0, 'max': 20},
            'Glucose': {'min': 0, 'max': 500},
            'BloodPressure': {'min': 0, 'max': 300},
            'SkinThickness': {'min': 0, 'max': 100},
            'Insulin': {'min': 0, 'max': 1000},
            'BMI': {'min': 0, 'max': 100},
            'DiabetesPedigreeFunction': {'min': 0, 'max': 3},
            'Age': {'min': 0, 'max': 120}
        }

        errors = []
        for feature, value in data.items():
            if not isinstance(value, (int, float)):
                errors.append(f"{feature} must be a number")
            elif value < validation_rules[feature]['min'] or value > validation_rules[feature]['max']:
                errors.append(f"{feature} must be between {validation_rules[feature]['min']} and {validation_rules[feature]['max']}")

        return errors

    def fit(self, X, y):
        """Train the model with provided data."""
        try:
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Calculate and store cross-validation score
            self.cv_score = np.mean(cross_val_score(self.model, X_scaled, y, cv=5))
            
            # Store feature importances
            self.feature_importances = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            self.is_fitted = True
            
        except Exception as e:
            raise RuntimeError(f"Error during model training: {str(e)}")

    def get_risk_factors(self, input_data):
        """Analyze which features contribute most to the prediction."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before analyzing risk factors")

        scaled_input = self.scaler.transform(input_data)
        feature_contributions = []
        
        for i, (feature, importance) in enumerate(self.feature_importances.items()):
            value = input_data.iloc[0, i]
            contribution = importance * scaled_input[0][i]
            feature_contributions.append({
                'feature': feature,
                'value': value,
                'importance': importance,
                'contribution': contribution
            })
        
        return sorted(feature_contributions, key=lambda x: abs(x['contribution']), reverse=True)

    def predict_diabetes_risk(self, **kwargs):
        """
        Predict diabetes risk with detailed analysis.
        
        Args:
            **kwargs: Feature values as keyword arguments
        
        Returns:
            dict: Detailed prediction results and analysis
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Validate inputs
        input_data = {
            'Pregnancies': kwargs.get('pregnancies'),
            'Glucose': kwargs.get('glucose'),
            'BloodPressure': kwargs.get('blood_pressure'),
            'SkinThickness': kwargs.get('skin_thickness'),
            'Insulin': kwargs.get('insulin'),
            'BMI': kwargs.get('bmi'),
            'DiabetesPedigreeFunction': kwargs.get('diabetes_pedigree_function'),
            'Age': kwargs.get('age')
        }

        validation_errors = self.validate_input(input_data)
        if validation_errors:
            return {
                'error': True,
                'message': 'Input validation failed',
                'details': validation_errors
            }

        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            prediction_proba = self.model.predict_proba(input_scaled)[0]
            
            # Get risk factors
            risk_factors = self.get_risk_factors(input_df)
            
            # Determine risk level and recommendations
            risk_percentage = prediction_proba[1] * 100
            if risk_percentage < 20:
                risk_level = "Low"
                stage = "No Diabetes Indicated"
            elif risk_percentage < 40:
                risk_level = "Moderate"
                stage = "Stage 1: Prediabetes (Borderline)"
            elif risk_percentage < 60:
                risk_level = "High"
                stage = "Stage 2: Early Diabetes"
            else:
                risk_level = "Very High"
                stage = "Stage 3: Advanced Diabetes"

            return {
                'error': False,
                'prediction': {
                    'is_diabetic': bool(prediction),
                    'risk_percentage': round(risk_percentage, 2),
                    'risk_level': risk_level,
                    'stage': stage
                },
                'risk_factors': risk_factors,
                'model_confidence': {
                    'cross_validation_score': round(self.cv_score * 100, 2),
                    'prediction_probability': round(max(prediction_proba) * 100, 2)
                }
            }

        except Exception as e:
            return {
                'error': True,
                'message': 'Prediction failed',
                'details': str(e)
            }

def main():
    """Example usage of the DiabetesRiskPredictor."""
    # Load your dataset
    df = pd.read_csv('diabetes_data.csv')
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']

    # Initialize and train the predictor
    predictor = DiabetesRiskPredictor()
    predictor.fit(X, y)

    # Example prediction
    result = predictor.predict_diabetes_risk(
        pregnancies=0,
        glucose=95,
        blood_pressure=80,
        skin_thickness=22,
        insulin=60,
        bmi=24.5,
        diabetes_pedigree_function=0.5,
        age=28
    )

    if not result['error']:
        print(f"\nDiabetes Risk Assessment:")
        print(f"Stage: {result['prediction']['stage']}")
        print(f"Risk Level: {result['prediction']['risk_level']}")
        print(f"Risk Percentage: {result['prediction']['risk_percentage']}%")
        
        print("\nTop Risk Factors:")
        for factor in result['risk_factors'][:3]:
            print(f"- {factor['feature']}: {factor['value']} (Contribution: {abs(factor['contribution']):.3f})")
        
        print("\nModel Confidence:")
        print(f"Cross-validation Score: {result['model_confidence']['cross_validation_score']}%")
        print(f"Prediction Probability: {result['model_confidence']['prediction_probability']}%")
    else:
        print(f"Error: {result['message']}")
        print("Details:", result['details'])

if __name__ == "__main__":
    main()