import streamlit as st
import pandas as pd
 # import plotly.express as px
import numpy as np
import pickle
import os

# Set page config first
st.set_page_config(
    page_title="Fitness Attendance Predictor", 
    page_icon="🏋️", 
    layout="wide"
)

class FitnessAttendancePredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'final_model.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at: {model_path}")
            self.model = None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None

    def preprocess_input(self, input_data):
        try:
            day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
            time_mapping = {"AM": 0, "PM": 1}
            
            input_data['day_of_week'] = day_mapping[input_data['day_of_week']]
            input_data['time'] = time_mapping[input_data['time']]
            
            input_processed = pd.get_dummies(pd.DataFrame([input_data]), 
                                         columns=['category', 'time'])
            
            expected_columns = [
                'months_as_member', 'weight', 'days_before', 'day_of_week',
                'category_Cycling', 'category_HIIT', 'category_Strength',
                'category_Yoga', 'category_unknown', 'time_PM'
            ]
            
            for col in expected_columns:
                if col not in input_processed.columns:
                    input_processed[col] = 0
            
            return input_processed[expected_columns]
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return None

    def create_sidebar(self):
        with st.sidebar:
            st.header("🏋️ Member Details")
            
            months_as_member = st.slider(
                "Membership Duration (months)", 
                min_value=1, 
                max_value=60, 
                value=12,
                help="How long have you been a member?"
            )
            
            weight = st.slider(
                "Weight (kg)", 
                min_value=40.0, 
                max_value=150.0, 
                value=70.0, 
                step=0.5,
                help="Your current weight"
            )
            
            days_before = st.slider(
                "Days Before Booking", 
                min_value=1, 
                max_value=30, 
                value=7,
                help="How many days in advance are you booking?"
            )
            
            day_of_week = st.selectbox(
                "Preferred Class Day", 
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                help="Select the day of the week"
            )
            
            time = st.radio(
                "Preferred Class Time", 
                ["AM", "PM"],
                help="Morning or evening preference"
            )
            
            category = st.selectbox(
                "Class Category", 
                ["HIIT", "Cycling", "Strength", "Yoga", "Aqua", "unknown"],
                help="Type of fitness class"
            )
            
            return {
                'months_as_member': months_as_member,
                'weight': weight,
                'days_before': days_before,
                'day_of_week': day_of_week,
                'time': time,
                'category': category
            }

    def predict_attendance(self, input_data):
        if self.model is None:
            return None, None
            
        input_processed = self.preprocess_input(input_data)
        if input_processed is None:
            return None, None
            
        try:
            prediction = self.model.predict(input_processed)
            probability = self.model.predict_proba(input_processed)[0][1]
            return prediction[0], probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

    def create_insights_dashboard(self):
        st.header("📊 Insights Dashboard")
        
        tab1, tab2, tab3 = st.tabs([
            "Feature Importance", 
            "Prediction Distribution", 
            "Attendance Trends"
        ])
        
        with tab1:
            if hasattr(self.model, 'feature_importances_'):
                features = [
                    'Months as Member', 'Weight', 'Days Before', 'Day of Week',
                    'Cycling Class', 'HIIT Class', 'Strength Class', 
                    'Yoga Class', 'Unknown Class', 'PM Time'
                ]
                importances = self.model.feature_importances_
                
                fig = px.bar(
                    x=features, 
                    y=importances,
                    title="Feature Importance in Attendance Prediction",
                    labels={'x': 'Features', 'y': 'Importance'},
                    color=importances,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            probabilities = np.random.beta(5, 2, 1000)
            fig = px.histogram(
                x=probabilities,
                title="Distribution of Attendance Probabilities",
                labels={'x': 'Probability of Attendance'},
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            sample_data = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'Attendance Rate': [0.65, 0.72, 0.68, 0.75, 0.70, 0.73]
            })
            
            fig = px.line(
                sample_data,
                x='Month',
                y='Attendance Rate',
                title='Monthly Attendance Trends',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        st.title("🏋️‍♀️ Fitness Class Attendance Predictor")
        st.markdown("Predict your likelihood of attending fitness classes")
        
        input_data = self.create_sidebar()
        
        if st.sidebar.button("Predict Attendance", type="primary"):
            prediction, probability = self.predict_attendance(input_data)
            
            if prediction is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Attendance", 
                        "Will Attend" if prediction == 1 else "Will Not Attend",
                        delta="High Probability" if probability > 0.7 else "Low Probability"
                    )
                with col2:
                    st.metric(
                        "Attendance Probability", 
                        f"{probability:.2%}",
                        delta_color="inverse"
                    )
        
        self.create_insights_dashboard()
        
        st.markdown("---")
        with st.expander("About the Model"):
            st.markdown("""
            ### How the Prediction Works
            - Uses machine learning to predict fitness class attendance
            - Considers factors like membership duration, weight, booking timing
            - Provides probabilistic prediction of attendance
            """)

if __name__ == "__main__":
    predictor = FitnessAttendancePredictor()
    predictor.run()
