from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from datetime import datetime

app = Flask(__name__)

class DiabetesRiskPredictor:
    def __init__(self):
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.is_fitted = False
        
    def get_bmi_category(self, bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 24.9:
            return "Normal weight"
        elif bmi < 29.9:
            return "Overweight"
        else:
            return "Obese"
            
    def get_glucose_status(self, glucose):
        if glucose < 70:
            return "Low (Hypoglycemic)"
        elif glucose < 100:
            return "Normal"
        elif glucose < 126:
            return "Prediabetic"
        else:
            return "Diabetic range"
            
    def get_blood_pressure_status(self, bp):
        if bp < 90:
            return "Low"
        elif bp < 120:
            return "Normal"
        elif bp < 130:
            return "Elevated"
        elif bp < 140:
            return "High (Stage 1)"
        else:
            return "High (Stage 2)"

    def get_lifestyle_recommendations(self, risk_level, metrics):
        recommendations = {
            "Diet": [],
            "Exercise": [],
            "Monitoring": [],
            "Lifestyle": [],
            "Medical": []
        }
        
        # Diet recommendations
        if metrics['glucose'] > 100:
            recommendations["Diet"].extend([
                "Limit refined carbohydrates and sugary foods",
                "Increase fiber intake through vegetables and whole grains",
                "Include lean proteins in every meal",
                "Practice portion control"
            ])
            
        if metrics['bmi'] > 25:
            recommendations["Diet"].extend([
                "Create a modest calorie deficit",
                "Increase protein intake to preserve muscle mass",
                "Choose water over sugary beverages"
            ])
            
        # Exercise recommendations
        recommendations["Exercise"].extend([
            "Aim for 150 minutes of moderate exercise weekly",
            "Include both cardio and strength training",
            "Take regular walking breaks during the day"
        ])
        
        if metrics['blood_pressure'] > 120:
            recommendations["Exercise"].append("Focus on low-impact cardiovascular activities")
            
        # Monitoring recommendations based on risk level
        if risk_level in ["High", "Very High"]:
            recommendations["Monitoring"].extend([
                "Check blood glucose levels regularly",
                "Monitor blood pressure daily",
                "Keep a food and exercise diary",
                "Track your weight weekly"
            ])
        else:
            recommendations["Monitoring"].extend([
                "Regular health check-ups",
                "Annual blood work",
                "Periodic blood pressure checks"
            ])
            
        # Lifestyle recommendations
        recommendations["Lifestyle"].extend([
            "Ensure 7-9 hours of quality sleep",
            "Practice stress management techniques",
            "Maintain a consistent daily routine",
            "Stay hydrated with 8 glasses of water daily"
        ])
        
        # Medical recommendations based on risk
        if risk_level == "Very High":
            recommendations["Medical"].extend([
                "Schedule immediate consultation with healthcare provider",
                "Consider diabetes screening tests",
                "Discuss medication options with your doctor",
                "Regular foot examinations"
            ])
        elif risk_level == "High":
            recommendations["Medical"].extend([
                "Schedule check-up within next month",
                "Consider preventive medications",
                "Regular eye examinations"
            ])
        else:
            recommendations["Medical"].extend([
                "Annual medical check-up",
                "Regular preventive screenings"
            ])
            
        return recommendations

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.cv_score = np.mean(cross_val_score(self.model, X_scaled, y, cv=5))
        self.feature_importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        self.is_fitted = True

    def predict(self, input_data):
        if not self.is_fitted:
            return {"error": "Model not trained"}
            
        try:
            data = pd.DataFrame([input_data])
            scaled_data = self.scaler.transform(data)
            
            prediction = self.model.predict(scaled_data)[0]
            probabilities = self.model.predict_proba(scaled_data)[0]
            
            risk_percentage = probabilities[1] * 100
            
            # Determine risk level
            if risk_percentage < 20:
                risk_level = "Low"
                alert_class = "success"
            elif risk_percentage < 40:
                risk_level = "Moderate"
                alert_class = "warning"
            elif risk_percentage < 60:
                risk_level = "High"
                alert_class = "error"
            else:
                risk_level = "Very High"
                alert_class = "critical"

            # Get health metrics status
            metrics = {
                'bmi': input_data['BMI'],
                'glucose': input_data['Glucose'],
                'blood_pressure': input_data['BloodPressure']
            }
            
            # Get detailed recommendations
            recommendations = self.get_lifestyle_recommendations(risk_level, metrics)

            # Generate health insights
            health_insights = {
                "BMI Status": self.get_bmi_category(input_data['BMI']),
                "Glucose Level": self.get_glucose_status(input_data['Glucose']),
                "Blood Pressure": self.get_blood_pressure_status(input_data['BloodPressure'])
            }

            return {
                "success": True,
                "prediction": bool(prediction),
                "risk_percentage": round(risk_percentage, 1),
                "risk_level": risk_level,
                "alert_class": alert_class,
                "recommendations": recommendations,
                "health_insights": health_insights,
                "confidence_score": round(self.cv_score * 100, 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {"error": str(e)}

# Initialize predictor
predictor = DiabetesRiskPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Pregnancies': float(request.form['pregnancies']),
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['blood_pressure']),
            'SkinThickness': float(request.form['skin_thickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree_function']),
            'Age': float(request.form['age'])
        }
        
        result = predictor.predict(input_data)
        
        if "error" in result:
            return render_template('index.html', error=result["error"])
            
        return render_template('index.html', result=result)
        
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Load and prepare data
    df = pd.read_csv('diabetes_data.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Train the model
    predictor.fit(X, y)
    
    # Run the app
    app.run(debug=True)