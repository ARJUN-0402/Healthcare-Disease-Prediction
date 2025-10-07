import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

class HealthcarePredictionApp:
    """Streamlit web app for healthcare disease prediction"""

    def __init__(self):
        self.models_dir = 'models'
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all trained models"""
        try:
            # Load models for each disease
            diseases = ['diabetes', 'heart_disease', 'kidney_disease']
            model_types = ['xgboost', 'random_forest', 'svm']

            for disease in diseases:
                for model_type in model_types:
                    model_path = os.path.join(self.models_dir, f'{disease}_{model_type}.pkl')
                    if os.path.exists(model_path):
                        model_key = f"{disease}_{model_type}"
                        self.models[model_key] = joblib.load(model_path)

            print(f"Loaded {len(self.models)} models")

        except Exception as e:
            st.error(f"Error loading models: {e}")

    def get_input_form_diabetes(self):
        """Get input form for diabetes prediction"""
        st.subheader("Diabetes Risk Factors")

        col1, col2 = st.columns(2)

        with col1:
            num_pregnant = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=100)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)

        return {
            'num_pregnant': num_pregnant,
            'glucose_concentration': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'bmi': bmi,
            'diabetes_pedigree': diabetes_pedigree,
            'age': age
        }

    def get_input_form_heart_disease(self):
        """Get input form for heart disease prediction"""
        st.subheader("Heart Disease Risk Factors")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)

        with col2:
            chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
            thalach = st.number_input("Maximum Heart Rate", min_value=0, max_value=250, value=150)

        with col3:
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

        return {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

    def get_input_form_kidney_disease(self):
        """Get input form for kidney disease prediction"""
        st.subheader("Kidney Disease Risk Factors")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
            blood_urea = st.number_input("Blood Urea (mg/dL)", min_value=0.0, max_value=400.0, value=40.0, step=0.1)
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)

        with col2:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=500, value=100)

        return {
            'age': age, 'blood_pressure': blood_pressure, 'blood_urea': blood_urea,
            'serum_creatinine': serum_creatinine, 'hemoglobin': hemoglobin, 'glucose': glucose
        }

    def predict_disease(self, disease_type, input_data, selected_model):
        """Make prediction using selected model"""
        try:
            model_key = f"{disease_type}_{selected_model}"
            model = self.models.get(model_key)

            if model is None:
                return None, f"Model {model_key} not found"

            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            # Get prediction probability
            prob_positive = probability[1] if len(probability) > 1 else probability[0]

            return prediction, prob_positive

        except Exception as e:
            return None, f"Prediction error: {e}"

    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="Healthcare Disease Prediction",
            page_icon="🏥",
            layout="wide"
        )

        st.title("🏥 Healthcare Disease Prediction System")
        st.markdown("---")

        # Sidebar
        st.sidebar.title("Navigation")
        disease_type = st.sidebar.selectbox(
            "Select Disease Type",
            options=["diabetes", "heart_disease", "kidney_disease"],
            format_func=lambda x: {
                "diabetes": "🩸 Diabetes",
                "heart_disease": "❤️ Heart Disease",
                "kidney_disease": "🫘 Kidney Disease"
            }[x]
        )

        # Model selection
        available_models = [model for model in self.models.keys() if model.startswith(disease_type)]
        if available_models:
            model_names = [model.split('_')[1] for model in available_models]
            selected_model = st.sidebar.selectbox("Select Model", options=model_names)
        else:
            st.error(f"No models available for {disease_type}")
            return

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Instructions")
        st.sidebar.info(
            "Fill in the patient information and click 'Predict' to get disease risk assessment."
        )

        # Main content
        st.header(f"{'🩸 Diabetes' if disease_type == 'diabetes' else '❤️ Heart Disease' if disease_type == 'heart_disease' else '🫘 Kidney Disease'} Prediction")

        # Get input form based on disease type
        if disease_type == "diabetes":
            input_data = self.get_input_form_diabetes()
        elif disease_type == "heart_disease":
            input_data = self.get_input_form_heart_disease()
        else:
            input_data = self.get_input_form_kidney_disease()

        # Prediction button
        if st.button("🔮 Predict Disease Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                prediction, probability = self.predict_disease(disease_type, input_data, selected_model)

            if prediction is not None:
                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.error("⚠️ High Risk Detected!")
                        st.markdown(f"**Risk Probability:** {probability:.1%}")
                    else:
                        st.success("✅ Low Risk")
                        st.markdown(f"**Risk Probability:** {probability:.1%}")

                with col2:
                    # Risk meter
                    risk_percentage = probability * 100
                    st.metric(
                        label="Risk Score",
                        value=f"{risk_percentage:.1f}%",
                        delta="High Risk" if float(risk_percentage) > 50 else "Low Risk"
                    )

                # Additional information
                st.markdown("---")
                st.subheader("📊 Prediction Details")

                if prediction == 1:
                    st.warning("**Recommendation:** This patient shows signs of elevated disease risk. Please consult with healthcare professionals for further evaluation and testing.")
                else:
                    st.info("**Recommendation:** This patient appears to have low disease risk based on the provided information. Continue regular health monitoring.")

                # Model information
                model_name = selected_model.upper() if selected_model else "Unknown"
                st.caption(f"Prediction made using {model_name} model with {probability:.1%} confidence")

            else:
                st.error(f"Prediction failed: {probability}")

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>⚠️ This tool is for educational and research purposes only.</p>
                <p>Not intended for actual medical diagnosis or treatment decisions.</p>
                <p>Always consult qualified healthcare professionals for medical advice.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function"""
    app = HealthcarePredictionApp()
    app.run()

if __name__ == "__main__":
    main()