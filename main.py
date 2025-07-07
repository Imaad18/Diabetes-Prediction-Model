import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with improved sidebar visibility
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-color: #6366f1;
            --primary-light: #8b5cf6;
            --primary-dark: #4f46e5;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #06b6d4;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --border-color: #334155;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        /* Improved Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        [data-testid="stSidebar"] .st-cq {
            color: var(--text-primary) !important;
        }
        
        [data-testid="stSidebar"] .st-cr {
            background-color: rgba(99, 102, 241, 0.2) !important;
        }
        
        [data-testid="stSidebar"] .st-ck {
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Headers */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            color: var(--primary-light);
        }
        
        /* Buttons */
        .stButton>button {
            background: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }
        
        /* Prediction Results */
        .prediction-box {
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        
        .positive {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .negative {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Train and cache model
@st.cache_resource
def load_model(df):
    try:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Feature importance analysis
@st.cache_data
def get_feature_importance(df):
    model = load_model(df)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': df.drop('Outcome', axis=1).columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    return importance_df

# Data statistics
@st.cache_data
def get_data_stats(df):
    total_patients = len(df)
    diabetic_patients = df['Outcome'].sum()
    return {
        'total': total_patients,
        'diabetic': diabetic_patients,
        'non_diabetic': total_patients - diabetic_patients,
        'diabetic_percentage': (diabetic_patients / total_patients) * 100
    }

def show_prediction_page(model, df):
    st.markdown("## 🎯 Diabetes Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        
    with col2:
        insulin = st.slider("Insulin (mu U/ml)", 0, 850, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.4, step=0.01)
        age = st.slider("Age (years)", 20, 100, 30)
    
    if st.button("Predict Diabetes Risk", use_container_width=True):
        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]])
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100
        
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="prediction-box positive">
                <h3>⚠️ High Diabetes Risk</h3>
                <p>Probability: <strong>{probability:.1f}%</strong></p>
                <p>This patient shows significant risk factors for diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box negative">
                <h3>✅ Low Diabetes Risk</h3>
                <p>Probability: <strong>{probability:.1f}%</strong></p>
                <p>This patient shows no significant risk factors for diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "#10b981"},
                    {'range': [30, 70], 'color': "#f59e0b"},
                    {'range': [70, 100], 'color': "#ef4444"}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': probability}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

def show_data_analysis_page(df, stats):
    st.markdown("## 📊 Dataset Analysis")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{stats['total']:,}")
    with col2:
        st.metric("Diabetic Patients", f"{stats['diabetic']:,}", f"{stats['diabetic_percentage']:.1f}%")
    with col3:
        st.metric("Non-Diabetic", f"{stats['non_diabetic']:,}")
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Distribution plots
    st.markdown("### Feature Distributions")
    selected_feature = st.selectbox("Select feature", df.columns[:-1])
    
    tab1, tab2 = st.tabs(["Histogram", "Box Plot"])
    
    with tab1:
        fig = px.histogram(df, x=selected_feature, color='Outcome', nbins=30,
                          color_discrete_map={0: "#10b981", 1: "#ef4444"})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.box(df, x='Outcome', y=selected_feature, color='Outcome',
                    color_discrete_map={0: "#10b981", 1: "#ef4444"})
        st.plotly_chart(fig, use_container_width=True)

def show_model_insights_page(model, df):
    st.markdown("## 🔍 Model Insights")
    
    # Split data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{accuracy_score(y_train, train_preds):.1%}")
    with col2:
        st.metric("Test Accuracy", f"{accuracy_score(y_test, test_preds):.1%}")
    
    # Confusion matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, test_preds)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual"),
                   x=['No Diabetes', 'Diabetes'],
                   y=['No Diabetes', 'Diabetes'],
                   text_auto=True,
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### Feature Importance")
    importance_df = get_feature_importance(df)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This application uses machine learning to predict diabetes risk based on health metrics.
    
    ### Features:
    - Random Forest Classifier
    - Interactive visualizations
    - Feature importance analysis
    - Probability scoring
    
    ### Data Requirements:
    The dataset must contain these columns:
    - Pregnancies
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - DiabetesPedigreeFunction
    - Age
    - Outcome (0/1)
    
    ### Disclaimer:
    This tool is for educational purposes only and not a substitute for professional medical advice.
    """)

def show_welcome_page():
    st.markdown("""
    <div class="main-header">🩺 Diabetes Prediction App</div>
    <p style="text-align: center; color: var(--text-secondary); margin-bottom: 2rem;">
        Advanced ML-powered diabetes risk assessment
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 How to use:
    1. Upload a CSV file with diabetes data using the sidebar
    2. Navigate between different sections:
       - **🎯 Prediction**: Make individual predictions
       - **📊 Data Analysis**: Explore the dataset
       - **🔍 Model Insights**: View model performance
       - **ℹ️ About**: Learn about the app
    
    ### Sample Data Format:
    """)
    
    sample_data = {
        'Pregnancies': [6, 1],
        'Glucose': [148, 85],
        'BloodPressure': [72, 66],
        'SkinThickness': [35, 29],
        'Insulin': [0, 0],
        'BMI': [33.6, 26.6],
        'DiabetesPedigreeFunction': [0.627, 0.351],
        'Age': [50, 31],
        'Outcome': [1, 0]
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

def main():
    inject_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 1rem; margin-bottom: 1rem;">
            <h2>🔬 Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    # Main content
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            model = load_model(df)
            stats = get_data_stats(df)
            
            with st.sidebar:
                st.success("✅ Data loaded successfully")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Dataset Stats</h4>
                    <p>Total: {stats['total']:,}</p>
                    <p>Diabetic: {stats['diabetic']:,}</p>
                    <p>Non-diabetic: {stats['non_diabetic']:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                page = st.selectbox(
                    "Navigate to",
                    ["🎯 Prediction", "📊 Data Analysis", "🔍 Model Insights", "ℹ️ About"],
                    key="navigation"
                )
            
            if page == "🎯 Prediction":
                show_prediction_page(model, df)
            elif page == "📊 Data Analysis":
                show_data_analysis_page(df, stats)
            elif page == "🔍 Model Insights":
                show_model_insights_page(model, df)
            elif page == "ℹ️ About":
                show_about_page()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        show_welcome_page()

if __name__ == "__main__":
    main()
    
