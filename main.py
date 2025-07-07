import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Variables */
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
            --shadow-lg: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-success: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-danger: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        /* Dark theme setup */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Enhanced Header */
        .main-header {
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
        }
        
        .main-header::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: var(--gradient-primary);
            border-radius: 2px;
        }
        
        /* Advanced Card Styling */
        .metric-card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow-lg);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--gradient-primary);
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        /* Prediction Result Cards */
        .prediction-positive {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        
        .prediction-positive::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #ef4444, #dc2626);
        }
        
        .prediction-negative {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        
        .prediction-negative::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #10b981, #059669);
        }
        
        /* Info Boxes */
        .info-box {
            background: rgba(6, 182, 212, 0.1);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
        }
        
        .warning-box {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
        }
        
        .success-box {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
        }
        
        /* Sidebar Enhancements */
        .css-1d391kg, .css-1cypcdb, .css-17eq0hr {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .css-1d391kg .css-1v0mbdj {
            background: transparent !important;
        }
        
        /* Button Styling */
        .stButton > button {
            background: var(--gradient-primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2) !important;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        /* Metrics Enhancement */
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow-lg) !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-xl) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        
        div[data-testid="metric-container"] > div {
            color: var(--text-primary) !important;
        }
        
        div[data-testid="metric-container"] label {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }
        
        /* Slider Enhancements */
        .stSlider > div > div > div > div {
            background: var(--gradient-primary) !important;
        }
        
        .stSlider > div > div > div[role="slider"] {
            background: white !important;
            border: 3px solid var(--primary-color) !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Selectbox Styling */
        .stSelectbox > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
        }
        
        /* File Uploader */
        .stFileUploader > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 2px dashed var(--primary-color) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stFileUploader > div > div:hover {
            border-color: var(--primary-light) !important;
            background: rgba(30, 41, 59, 0.9) !important;
        }
        
        /* Dataframe Styling */
        .stDataFrame {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
        }
        
        .stDataFrame table {
            background: transparent !important;
            color: var(--text-primary) !important;
        }
        
        .stDataFrame th {
            background: rgba(99, 102, 241, 0.1) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        
        .stDataFrame td {
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        
        /* Multiselect */
        .stMultiSelect > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
        }
        
        .stMultiSelect span {
            color: var(--text-primary) !important;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background: var(--gradient-primary) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(30, 41, 59, 0.8) !important;
            border-radius: 12px !important;
            padding: 0.5rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
        }
        
        /* Plotly charts background */
        .js-plotly-plot {
            background: rgba(30, 41, 59, 0.4) !important;
            border-radius: 16px !important;
            padding: 1rem !important;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .metric-card, .prediction-positive, .prediction-negative {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gradient-primary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-dark);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2.5rem;
            }
            
            .metric-card, .prediction-positive, .prediction-negative {
                padding: 1rem;
            }
        }
        
        /* Loading animations */
        .loading-spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Enhanced tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover::after {
            opacity: 1;
            visibility: visible;
        }
        
        /* Floating elements */
        .float-element {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--gradient-primary);
            color: white;
            padding: 1rem;
            border-radius: 50%;
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        /* Enhanced form styling */
        .stTextInput > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
        }
        
        .stTextInput > div > div:focus-within {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
        }
        
        /* Enhanced navigation */
        .nav-item {
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .nav-item:hover {
            background: rgba(99, 102, 241, 0.1);
            transform: translateX(5px);
        }
        
        .nav-item.active {
            background: var(--gradient-primary);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Enhanced theme colors for plots
PLOT_THEME = {
    'background': 'rgba(15, 23, 42, 0.8)',
    'paper_bgcolor': 'rgba(15, 23, 42, 0.8)',
    'plot_bgcolor': 'rgba(30, 41, 59, 0.4)',
    'font_color': '#f8fafc',
    'grid_color': 'rgba(255, 255, 255, 0.1)',
    'primary_color': '#6366f1',
    'success_color': '#10b981',
    'danger_color': '#ef4444',
    'warning_color': '#f59e0b',
}

# Load and prepare data
@st.cache_data
def load_data(uploaded_file):
    """Load the diabetes dataset from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.stop()
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Train and cache model
@st.cache_resource
def load_model(df):
    """Load or train the diabetes prediction model"""
    try:
        # Try to load a pre-trained model
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # If no model exists, train a new one
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
        
        try:
            with open('diabetes_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        except:
            pass
        
        return model

# Feature importance analysis
@st.cache_data
def get_feature_importance(df):
    """Get feature importance from the model"""
    model = load_model(df)
    feature_names = df.drop('Outcome', axis=1).columns
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# Data statistics
@st.cache_data
def get_data_stats(df):
    """Get basic statistics about the dataset"""
    total_patients = len(df)
    diabetic_patients = df['Outcome'].sum()
    non_diabetic_patients = total_patients - diabetic_patients
    
    return {
        'total': total_patients,
        'diabetic': diabetic_patients,
        'non_diabetic': non_diabetic_patients,
        'diabetic_percentage': (diabetic_patients / total_patients) * 100
    }

# Enhanced plotting functions
def create_styled_plot(fig, title=""):
    """Apply consistent styling to plots"""
    fig.update_layout(
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=dict(color=PLOT_THEME['font_color'], family="Inter"),
        title=dict(
            text=title,
            font=dict(size=20, weight='bold'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor=PLOT_THEME['grid_color'],
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=PLOT_THEME['grid_color'],
            zeroline=False
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def show_prediction_page(model, df):
    """Show the prediction page with input form and results"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🎯 Diabetes Risk Assessment</h2>
        <p style="color: #cbd5e1;">Enter patient details to assess diabetes risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Patient Information")
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        
    with col2:
        st.markdown("### 📊 Health Metrics")
        insulin = st.slider("Insulin (mu U/ml)", 0, 850, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.4, step=0.01)
        age = st.slider("Age (years)", 20, 100, 30)
    
    # Prediction button
    if st.button("🔮 Predict Diabetes Risk", use_container_width=True):
        # Create input array
        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100
        
        # Display results
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="prediction-positive">
                <h2 style="color: #ef4444;">⚠️ High Diabetes Risk Detected</h2>
                <p style="font-size: 1.2rem;">Probability: <strong>{probability:.1f}%</strong></p>
                <p>This patient shows significant risk factors for diabetes. Consider recommending:</p>
                <ul>
                    <li>Further diagnostic tests (HbA1c, fasting glucose)</li>
                    <li>Lifestyle modifications (diet, exercise)</li>
                    <li>Consultation with an endocrinologist</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-negative">
                <h2 style="color: #10b981;">✅ Low Diabetes Risk</h2>
                <p style="font-size: 1.2rem;">Probability: <strong>{probability:.1f}%</strong></p>
                <p>This patient shows no significant risk factors for diabetes. Consider:</p>
                <ul>
                    <li>Regular health checkups</li>
                    <li>Maintaining healthy lifestyle</li>
                    <li>Periodic glucose monitoring if risk factors change</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability meter
        st.markdown("### 📈 Risk Probability")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6366f1"},
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
        
        fig = create_styled_plot(fig)
        st.plotly_chart(fig, use_container_width=True)

def show_data_analysis_page(df, stats):
    """Show data analysis and visualization page"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📊 Dataset Analysis</h2>
        <p style="color: #cbd5e1;">Explore the diabetes dataset characteristics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{stats['total']:,}")
    with col2:
        st.metric("Diabetic Patients", f"{stats['diabetic']:,}", f"{stats['diabetic_percentage']:.1f}%")
    with col3:
        st.metric("Non-Diabetic Patients", f"{stats['non_diabetic']:,}", 
                 f"{(100 - stats['diabetic_percentage']):.1f}%")
    
    # Show dataset sample
    st.markdown("### 📋 Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Distribution plots
    st.markdown("### 📈 Feature Distributions")
    
    tab1, tab2, tab3 = st.tabs(["📊 Numeric Features", "📈 Outcome Comparison", "🌡️ Correlation"])
    
    with tab1:
        selected_feature = st.selectbox(
            "Select a feature to visualize",
            df.drop('Outcome', axis=1).columns,
            key="feature_dist"
        )
        
        fig = px.histogram(
            df, 
            x=selected_feature, 
            nbins=30,
            marginal="box",
            title=f"Distribution of {selected_feature}",
            color_discrete_sequence=[PLOT_THEME['primary_color']]
        )
        fig = create_styled_plot(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        selected_feature = st.selectbox(
            "Select a feature to compare",
            df.drop('Outcome', axis=1).columns,
            key="feature_compare"
        )
        
        fig = px.box(
            df,
            x='Outcome',
            y=selected_feature,
            color='Outcome',
            title=f"{selected_feature} by Diabetes Outcome",
            color_discrete_map={0: PLOT_THEME['success_color'], 1: PLOT_THEME['danger_color']}
        )
        fig = create_styled_plot(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        corr = df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Feature Correlation Matrix"
        )
        fig = create_styled_plot(fig)
        st.plotly_chart(fig, use_container_width=True)

def show_model_insights_page(model, df):
    """Show model performance and feature importance"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🔍 Model Insights</h2>
        <p style="color: #cbd5e1;">Understand how the model makes predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Split data for evaluation
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model evaluation
    st.markdown("### 🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Train accuracy
        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        st.metric("Training Accuracy", f"{train_acc:.1%}")
        
        # Confusion matrix
        st.markdown("#### Confusion Matrix (Test Set)")
        test_preds = model.predict(X_test)
        cm = confusion_matrix(y_test, test_preds)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Diabetes', 'Diabetes'],
            y=['No Diabetes', 'Diabetes'],
            color_continuous_scale='Blues'
        )
        fig = create_styled_plot(fig, "Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Test accuracy
        test_acc = accuracy_score(y_test, test_preds)
        st.metric("Test Accuracy", f"{test_acc:.1%}")
        
        # Classification report
        st.markdown("#### Classification Report (Test Set)")
        report = classification_report(y_test, test_preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    # Feature importance
    st.markdown("### 📊 Feature Importance")
    importance_df = get_feature_importance(df)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance Scores",
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig = create_styled_plot(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision path explanation
    st.markdown("### 🤖 How the Model Makes Decisions")
    st.markdown("""
    The Random Forest model makes predictions by:
    
    1. **Analyzing multiple decision trees**: Each tree votes on the outcome
    2. **Considering feature importance**: More important features have greater influence
    3. **Averaging predictions**: The final prediction is based on majority vote
    
    Key factors in diabetes prediction:
    - Glucose levels are typically the most important predictor
    - BMI and Age also contribute significantly
    - Other factors provide additional context
    """)

def show_about_page():
    """Show information about the app"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>ℹ️ About This App</h2>
        <p style="color: #cbd5e1;">Learn about the diabetes prediction model</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Diabetes Prediction App
    
    This application uses machine learning to predict the likelihood of diabetes based on patient health metrics.
    
    ### 🔧 How It Works
    
    1. **Data Input**: Users can upload a CSV file with patient health data
    2. **Model Prediction**: The app uses a trained Random Forest classifier
    3. **Results Visualization**: Predictions are displayed with probabilities and explanations
    
    ### 🧠 Model Details
    
    - **Algorithm**: Random Forest Classifier
    - **Features Used**: 8 health metrics (Glucose, BMI, Age, etc.)
    - **Accuracy**: ~75-80% on test data (varies by dataset)
    
    ### ⚠️ Important Notes
    
    - This tool is for **educational purposes only**
    - Not a substitute for professional medical advice
    - Always consult healthcare professionals for medical decisions
    
    ### 📚 References
    
    - [Diabetes Dataset Source](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
    - [Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    
    ### 👨‍💻 Developer
    
    This app was developed using:
    - Python 3
    - Streamlit
    - Scikit-learn
    - Plotly
    
    For questions or feedback, please contact the developer.
    """)

# Main app
def main():
    inject_css()
    
    # Enhanced header with animation
    st.markdown('''
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🩺 Diabetes Prediction App</h1>
        <p style="color: #cbd5e1; font-size: 1.2rem; margin-top: 1rem;">
            Advanced ML-powered diabetes risk assessment
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <h2 style="color: #6366f1; font-weight: 600;">🔬 Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a diabetes dataset CSV file with the required columns"
        )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        model = load_model(df)
        stats = get_data_stats(df)
        
        with st.sidebar:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Dataset loaded successfully!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Dataset Overview</h4>
                <p><strong>{len(df):,}</strong> total records</p>
                <p><strong>{stats['diabetic']:,}</strong> diabetic cases</p>
                <p><strong>{stats['diabetic_percentage']:.1f}%</strong> diabetes rate</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🧭 Navigation")
            page = st.selectbox(
                "Choose a page",
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
    else:
        show_welcome_page()

if __name__ == "__main__":
    main()
