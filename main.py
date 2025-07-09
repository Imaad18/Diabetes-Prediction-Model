import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
import io
import base64
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --primary-color: #6366f1;
            --primary-light: #8b5cf6;
            --primary-dark: #4f46e5;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #06b6d4;
            --dark-bg: #0f172a;
            --darker-bg: #020617;
            --card-bg: #1e293b;
            --card-hover: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border-color: #334155;
            --border-light: rgba(255, 255, 255, 0.1);
            --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        
        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(ellipse at top, #1e293b 0%, #0f172a 50%, #020617 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: transparent;
        }
        
        /* Enhanced Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
            border-right: 1px solid var(--border-light) !important;
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: rgba(51, 65, 85, 0.8) !important;
            border: 1px solid var(--border-light) !important;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            background: var(--primary-color) !important;
            border: none !important;
            color: white !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--primary-dark) !important;
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }
        
        /* Enhanced Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%);
            border: 1px solid var(--border-light);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
            border-color: var(--primary-color);
        }
        
        .feature-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            background: linear-gradient(135deg, rgba(51, 65, 85, 0.8) 0%, rgba(71, 85, 105, 0.6) 100%);
            border-color: var(--primary-light);
        }
        
        /* Headers */
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            background: linear-gradient(135deg, var(--primary-light), var(--info-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            color: var(--primary-light);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-dark), var(--primary-color)) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Prediction Results */
        .prediction-box {
            padding: 2rem;
            border-radius: 16px;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            animation: fadeInUp 0.5s ease;
        }
        
        .positive {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
            border: 2px solid rgba(239, 68, 68, 0.4);
        }
        
        .negative {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
            border: 2px solid rgba(16, 185, 129, 0.4);
        }
        
        .warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%);
            border: 2px solid rgba(245, 158, 11, 0.4);
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeInUp 0.6s ease;
        }
        
        /* Metrics */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.3);
        }
        
        /* Enhanced Data Display */
        .stDataFrame {
            background: rgba(30, 41, 59, 0.8) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-light) !important;
            overflow: hidden !important;
        }
        
        .stDataFrame [data-testid="stTable"] {
            background: transparent !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            padding: 0.25rem;
            border: 1px solid var(--border-light);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary-light);
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--primary-color) !important;
            color: white !important;
        }
        
        /* Sliders */
        .stSlider > div > div > div {
            background: var(--primary-color) !important;
        }
        
        .stSlider > div > div > div > div {
            background: var(--primary-light) !important;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.8);
            border: 2px dashed var(--border-light);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary-color);
            background: rgba(99, 102, 241, 0.1);
        }
        
        /* Code blocks */
        .stCode {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 8px !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            white-space: nowrap;
            z-index: 1000;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
        }
        
        /* Loading states */
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        .loading::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 20px;
            height: 20px;
            border: 2px solid var(--primary-color);
            border-top: 2px solid transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: translate(-50%, -50%) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg); }
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary-color), var(--primary-light)) !important;
        }
        
        /* Expander */
        .stExpander {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 12px !important;
        }
        
        .stExpander > div > div {
            background: transparent !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-light);
        }
    </style>
    """, unsafe_allow_html=True)

# Enhanced data validation
def validate_data(df):
    """Comprehensive data validation with detailed feedback"""
    issues = []
    warnings = []
    
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and ranges
    if 'Glucose' in df.columns:
        if df['Glucose'].min() < 0 or df['Glucose'].max() > 300:
            warnings.append("Glucose values outside normal range (0-300)")
    
    if 'BloodPressure' in df.columns:
        if df['BloodPressure'].min() < 0 or df['BloodPressure'].max() > 200:
            warnings.append("Blood pressure values outside normal range (0-200)")
    
    if 'BMI' in df.columns:
        if df['BMI'].min() < 10 or df['BMI'].max() > 70:
            warnings.append("BMI values outside normal range (10-70)")
    
    if 'Age' in df.columns:
        if df['Age'].min() < 0 or df['Age'].max() > 120:
            warnings.append("Age values outside normal range (0-120)")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        warnings.append(f"Missing values detected: {missing_values[missing_values > 0].to_dict()}")
    
    # Check outcome distribution
    if 'Outcome' in df.columns:
        outcome_dist = df['Outcome'].value_counts()
        if len(outcome_dist) != 2:
            issues.append("Outcome column must contain exactly 2 classes (0 and 1)")
        elif outcome_dist.min() / outcome_dist.max() < 0.1:
            warnings.append("Severe class imbalance detected")
    
    return issues, warnings

# Enhanced data loading with validation
@st.cache_data
def load_data(uploaded_file):
    """Load data with comprehensive validation and preprocessing"""
    try:
        # Read the file
        df = pd.read_csv(uploaded_file)
        
        # Validate data
        issues, warnings = validate_data(df)
        
        if issues:
            st.error("❌ Data validation failed:")
            for issue in issues:
                st.error(f"• {issue}")
            st.stop()
        
        if warnings:
            st.warning("⚠️ Data validation warnings:")
            for warning in warnings:
                st.warning(f"• {warning}")
        
        # Basic preprocessing
        df = df.dropna()  # Remove rows with missing values
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            st.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Ensure proper data types
        numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Outcome'] = df['Outcome'].astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Enhanced model training with multiple algorithms
@st.cache_resource
def train_models(df):
    """Train multiple models and return the best one"""
    try:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }
        
        # Train and evaluate models
        model_results = {}
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                # Use scaled data for these models
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = accuracy_score(y_test, y_pred)
            else:
                # Use original data for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'score': score,
                'predictions': y_pred,
                'needs_scaling': name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']
            }
            
            if score > best_score:
                best_score = score
                best_model = name
        
        return model_results, best_model, scaler
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        st.stop()

# Enhanced feature importance with SHAP-like analysis
@st.cache_data
def get_comprehensive_feature_analysis(df, model_results):
    """Get comprehensive feature analysis"""
    try:
        # Get feature importance from Random Forest
        rf_model = model_results['Random Forest']['model']
        feature_names = df.drop('Outcome', axis=1).columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_,
            'Rank': range(1, len(feature_names) + 1)
        }).sort_values('Importance', ascending=False)
        
        # Calculate correlation with outcome
        correlations = df.corr()['Outcome'].drop('Outcome').abs().sort_values(ascending=False)
        
        # Combine importance and correlation
        analysis_df = importance_df.merge(
            correlations.reset_index().rename(columns={'index': 'Feature', 'Outcome': 'Correlation'}),
            on='Feature'
        )
        
        return analysis_df
        
    except Exception as e:
        st.error(f"Error in feature analysis: {str(e)}")
        return pd.DataFrame()

# Enhanced statistics with more insights
@st.cache_data
def get_comprehensive_stats(df):
    """Get comprehensive dataset statistics"""
    try:
        total_patients = len(df)
        diabetic_patients = df['Outcome'].sum()
        non_diabetic_patients = total_patients - diabetic_patients
        
        # Basic stats
        stats = {
            'total': total_patients,
            'diabetic': diabetic_patients,
            'non_diabetic': non_diabetic_patients,
            'diabetic_percentage': (diabetic_patients / total_patients) * 100,
            'balance_ratio': min(diabetic_patients, non_diabetic_patients) / max(diabetic_patients, non_diabetic_patients)
        }
        
        # Feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Outcome')
        feature_stats = {}
        
        for col in numeric_cols:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skewness': df[col].skew()
            }
        
        stats['features'] = feature_stats
        
        return stats
        
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return {}

# Enhanced prediction with confidence intervals
def make_enhanced_prediction(models, scaler, input_data, feature_names):
    """Make prediction with multiple models and confidence analysis"""
    try:
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        
        predictions = {}
        probabilities = {}
        
        for name, model_info in models.items():
            model = model_info['model']
            
            if model_info['needs_scaling']:
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]
            else:
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
            
            predictions[name] = pred
            probabilities[name] = prob
        
        # Calculate ensemble prediction
        avg_probability = np.mean(list(probabilities.values()))
        ensemble_prediction = 1 if avg_probability > 0.5 else 0
        
        # Calculate confidence (std of probabilities)
        confidence = 1 - np.std(list(probabilities.values()))
        
        return {
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'ensemble_prediction': ensemble_prediction,
            'ensemble_probability': avg_probability,
            'confidence': confidence
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_risk_interpretation(probability, confidence):
    """Generate risk interpretation based on probability and confidence"""
    if confidence < 70:
        reliability = "Low model agreement - consider additional testing"
    elif confidence < 85:
        reliability = "Moderate model agreement - results are fairly reliable"
    else:
        reliability = "High model agreement - results are highly reliable"
    
    if probability < 30:
        interpretation = "Your health metrics suggest a low risk for diabetes. Continue maintaining healthy habits."
    elif probability < 70:
        interpretation = "You have moderate risk factors. Consider lifestyle modifications and regular monitoring."
    else:
        interpretation = "Multiple risk factors detected. Consult with healthcare provider for comprehensive evaluation."
    
    return f"{interpretation} {reliability}"

def generate_recommendations(probability, input_data, feature_names, risk_factors):
    """Generate personalized recommendations based on risk assessment"""
    recommendations = []
    
    # Map input data to features
    feature_dict = dict(zip(feature_names, input_data))
    
    # General recommendations based on risk level
    if probability < 30:
        recommendations.append({
            'title': '✅ Maintain Current Lifestyle',
            'description': 'Your risk is low. Continue with regular exercise, balanced diet, and routine health checkups.'
        })
    elif probability < 70:
        recommendations.append({
            'title': '⚠️ Moderate Risk - Take Action',
            'description': 'Consider lifestyle modifications to reduce risk. Focus on diet, exercise, and weight management.'
        })
    else:
        recommendations.append({
            'title': '🔴 High Risk - Immediate Action Required',
            'description': 'Strongly recommend consultation with healthcare provider. Immediate lifestyle changes needed.'
        })
    
    # Specific recommendations based on individual metrics
    if feature_dict['BMI'] > 30:
        recommendations.append({
            'title': '🏃 Weight Management Priority',
            'description': 'Focus on gradual weight loss through caloric reduction and increased physical activity. Target BMI < 25.'
        })
    
    if feature_dict['Glucose'] > 140:
        recommendations.append({
            'title': '🍎 Blood Sugar Control',
            'description': 'Monitor carbohydrate intake, consider smaller frequent meals, and increase fiber consumption.'
        })
    
    if feature_dict['BloodPressure'] > 90:
        recommendations.append({
            'title': '💓 Blood Pressure Management',
            'description': 'Reduce sodium intake, increase potassium-rich foods, manage stress, and consider regular monitoring.'
        })
    
    if feature_dict['Age'] > 45:
        recommendations.append({
            'title': '🕐 Age-Related Prevention',
            'description': 'Increase screening frequency, focus on strength training, and maintain social connections.'
        })
    
    # Always include general health recommendations
    recommendations.append({
        'title': '🌟 General Health Optimization',
        'description': 'Maintain 7-9 hours of sleep, stay hydrated, manage stress, and schedule regular health screenings.'
    })
    
    return recommendations

def simulate_risk_timeline(input_data, models, scaler, feature_names):
    """Simulate risk progression over time"""
    timeline_data = []
    
    for years in range(0, 11):
        # Simulate aging and potential health changes
        modified_data = input_data.copy()
        modified_data[7] += years  # Age increases
        
        # Simulate gradual health decline (conservative estimates)
        if years > 0:
            modified_data[1] += years * 0.5  # Slight glucose increase
            modified_data[5] += years * 0.1   # Slight BMI increase
            modified_data[2] += years * 0.3   # Slight BP increase
        
        # Make prediction
        result = make_enhanced_prediction(models, scaler, modified_data, feature_names)
        if result:
            timeline_data.append({
                'Years': years,
                'Risk_Probability': result['ensemble_probability'] * 100
            })
    
    return pd.DataFrame(timeline_data)

def detect_outliers(df):
    """Detect outliers using IQR method"""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != 'Outcome':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            if outlier_count > 0:
                outliers[col] = outlier_count
    
    return outliers

# Enhanced prediction page
def show_enhanced_prediction_page(models, scaler, df, best_model_name):
    """Enhanced prediction page with better UX"""
    st.markdown('<div class="section-header">🎯 Advanced Diabetes Risk Assessment</div>', unsafe_allow_html=True)
    
    # Add explanation
    with st.expander("ℹ️ How to use this assessment", expanded=False):
        st.markdown("""
        This tool uses multiple machine learning models to assess diabetes risk based on key health indicators.
        
        **Key Features:**
        - Multiple ML algorithms for robust predictions
        - Confidence scoring for prediction reliability
        - Risk categorization (Low, Medium, High)
        - Personalized recommendations
        
        **Instructions:**
        1. Adjust the sliders to match your health metrics
        2. Click "Analyze Risk" to get your assessment
        3. Review the detailed results and recommendations
        """)
    
    # Input section with better organization
    st.markdown("### 📊 Health Metrics Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Reproductive Health**")
        pregnancies = st.slider(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=1,
            help="Total number of pregnancies"
        )
        
        st.markdown("**Glucose & Metabolism**")
        glucose = st.slider(
            "Glucose Level (mg/dL)",
            min_value=0, max_value=200, value=100,
            help="Plasma glucose concentration after 2-hour oral glucose tolerance test"
        )
        
        insulin = st.slider(
            "Insulin Level (mu U/ml)",
            min_value=0, max_value=850, value=80,
            help="2-hour serum insulin level"
        )
        
    with col2:
        st.markdown("**Physical Measurements**")
        blood_pressure = st.slider(
            "Blood Pressure (mm Hg)",
            min_value=0, max_value=130, value=70,
            help="Diastolic blood pressure"
        )
        
        skin_thickness = st.slider(
            "Skin Thickness (mm)",
            min_value=0, max_value=100, value=20,
            help="Triceps skin fold thickness"
        )
        
        bmi = st.slider(
            "BMI (Body Mass Index)",
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Weight in kg / (height in m)²"
        )
        
    with col3:
        st.markdown("**Genetic & Age Factors**")
        diabetes_pedigree = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=2.5, value=0.4, step=0.01,
            help="Diabetes heredity risk score"
        )
        
        age = st.slider(
            "Age (years)",
            min_value=20, max_value=100, value=30,
            help="Age in years"
        )
    
    # BMI category display
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "info"
    elif bmi < 25:
        bmi_category = "Normal"
        bmi_color = "success"
    elif bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "warning"
    else:
        bmi_category = "Obese"
        bmi_color = "danger"
    
    st.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")
    
    # Risk factor warnings
    risk_factors = []
    if glucose > 140:
        risk_factors.append("🔴 High glucose level")
    if blood_pressure > 90:
        risk_factors.append("🟠 Elevated blood pressure")
    if bmi > 30:
        risk_factors.append("🟠 High BMI")
    if age > 45:
        risk_factors.append("🟡 Age-related risk")
    
    if risk_factors:
        st.markdown("**⚠️ Risk Factors Detected:**")
        for factor in risk_factors:
            st.markdown(f"• {factor}")
    
    # Prediction button
    if st.button("🔍 Analyze Diabetes Risk", use_container_width=True):
        with st.spinner("Analyzing your health data..."):
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, diabetes_pedigree, age]
            
            feature_names = df.drop('Outcome', axis=1).columns
            
            result = make_enhanced_prediction(models, scaler, input_data, feature_names)
            
            if result:
                # Main prediction result
                ensemble_prob = result['ensemble_probability'] * 100
                confidence = result['confidence'] * 100
                
                # Determine risk category
                if ensemble_prob < 30:
                    risk_category = "Low"
                    risk_class = "negative"
                    risk_icon = "✅"
                elif ensemble_prob < 70:
                    risk_category = "Medium"
                    risk_class = "warning"
                    risk_icon = "⚠️"
                else:
                    risk_category = "High"
                    risk_class = "positive"
                    risk_icon = "🔴"
                
                # Display main result
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h2>{risk_icon} {risk_category} Diabetes Risk</h2>
                    <h3>Risk Probability: {ensemble_prob:.1f}%</h3>
                    <p>Confidence Level: {confidence:.1f}%</p>
                    <p style="margin-top: 1rem; font-size: 0.9em;">
                        {get_risk_interpretation(ensemble_prob, confidence)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=ensemble_prob,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Probability (%)", 'font': {'size': 20}},
                        delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#6366f1"},
                            'bgcolor': "rgba(255,255,255,0.1)",
                            'borderwidth': 2,
                            'bordercolor': "white",
                            'steps': [
                                {'range': [0, 30], 'color': "#10b981"},
                                {'range': [30, 70], 'color': "#f59e0b"},
                                {'range': [70, 100], 'color': "#ef4444"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': ensemble_prob
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "white", 'family': "Inter"}
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Model comparison
                    st.markdown("### 🤖 Model Consensus")
                    model_df = pd.DataFrame({
                        'Model': list(result['individual_probabilities'].keys()),
                        'Probability': [p * 100 for p in result['individual_probabilities'].values()],
                        'Prediction': ['High Risk' if p > 0.5 else 'Low Risk' for p in result['individual_probabilities'].values()]
                    })
                    
                    fig_models = px.bar(
                        model_df, 
                        x='Probability', 
                        y='Model', 
                        color='Probability',
                        orientation='h',
                        color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
                        title="Individual Model Predictions"
                    )
                    fig_models.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "white", 'family': "Inter"}
                    )
                    st.plotly_chart(fig_models, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Personalized Recommendations")
                recommendations = generate_recommendations(
                    ensemble_prob, input_data, feature_names, risk_factors
                )
                
                for i, rec in enumerate(recommendations):
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>{rec['title']}</h4>
                        <p>{rec['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk timeline
                st.markdown("### 📈 Risk Progression Timeline")
                timeline_data = simulate_risk_timeline(input_data, models, scaler, feature_names)
                
                fig_timeline = px.line(
                    timeline_data,
                    x='Years',
                    y='Risk_Probability',
                    title='Projected Risk Over Time',
                    labels={'Risk_Probability': 'Risk Probability (%)', 'Years': 'Years from Now'}
                )
                fig_timeline.add_hline(y=50, line_dash="dash", line_color="orange", 
                                     annotation_text="Moderate Risk Threshold")
                fig_timeline.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'family': "Inter"}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

# Enhanced data analysis page
def show_enhanced_data_analysis_page(df, stats, feature_analysis):
    """Enhanced data analysis with comprehensive visualizations"""
    st.markdown('<div class="section-header">📊 Comprehensive Dataset Analysis</div>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Patients", 
            f"{stats['total']:,}",
            help="Total number of patients in dataset"
        )
    
    with col2:
        st.metric(
            "Diabetic Cases", 
            f"{stats['diabetic']:,}",
            f"{stats['diabetic_percentage']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Non-Diabetic", 
            f"{stats['non_diabetic']:,}",
            f"{100-stats['diabetic_percentage']:.1f}%"
        )
    
    with col4:
        balance_status = "Balanced" if stats['balance_ratio'] > 0.7 else "Imbalanced"
        st.metric(
            "Class Balance", 
            f"{stats['balance_ratio']:.2f}",
            balance_status
        )
    
    # Data quality assessment
    st.markdown("### 🔍 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing values heatmap
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Feature",
                labels={'x': 'Features', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values detected")
    
    with col2:
        # Outlier detection
        outliers = detect_outliers(df)
        if outliers:
            st.markdown("**🎯 Outlier Detection:**")
            for feature, count in outliers.items():
                st.write(f"• {feature}: {count} outliers")
        else:
            st.success("✅ No significant outliers detected")
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    
    # Enhanced descriptive statistics
    desc_stats = df.describe()
    st.dataframe(desc_stats.round(2), use_container_width=True)
    
    # Feature distributions
    st.markdown("### 📊 Feature Distribution Analysis")
    
    # Feature selection
    feature_cols = df.columns[:-1].tolist()
    selected_features = st.multiselect(
        "Select features to analyze:",
        feature_cols,
        default=feature_cols[:4]
    )
    
    if selected_features:
        # Create subplot for multiple features
        n_features = len(selected_features)
        cols = min(2, n_features)
        rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=selected_features,
            specs=[[{"secondary_y": True}] * cols] * rows
        )
        
        for i, feature in enumerate(selected_features):
            row = i // cols + 1
            col = i % cols + 1
            
            # Histogram by outcome
            for outcome in [0, 1]:
                subset = df[df['Outcome'] == outcome][feature]
                fig.add_histogram(
                    x=subset,
                    name=f"Outcome {outcome}",
                    row=row, col=col,
                    opacity=0.7,
                    showlegend=(i == 0)
                )
        
        fig.update_layout(
            height=300 * rows,
            title_text="Feature Distributions by Outcome",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    # Correlation matrix
    corr_matrix = df.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"}
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature importance analysis
    st.markdown("### 🎯 Feature Importance Analysis")
    
    if not feature_analysis.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Importance ranking
            fig_importance = px.bar(
                feature_analysis,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance Ranking",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Importance vs correlation
            fig_scatter = px.scatter(
                feature_analysis,
                x='Correlation',
                y='Importance',
                text='Feature',
                title="Feature Importance vs Correlation",
                size='Importance',
                color='Importance',
                color_continuous_scale='plasma'
            )
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Advanced analytics
    st.markdown("### 🔬 Advanced Analytics")
    
    tabs = st.tabs(["📊 Distribution Analysis", "🎯 Risk Factor Analysis", "📈 Trend Analysis"])
    
    with tabs[0]:
        # Distribution analysis
        selected_feature = st.selectbox(
            "Select feature for detailed analysis:",
            df.columns[:-1]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig_box = px.box(
                df, 
                x='Outcome', 
                y=selected_feature,
                title=f"{selected_feature} Distribution by Outcome",
                color='Outcome',
                color_discrete_map={0: "#10b981", 1: "#ef4444"}
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Violin plot
            fig_violin = px.violin(
                df,
                x='Outcome',
                y=selected_feature,
                title=f"{selected_feature} Density by Outcome",
                color='Outcome',
                color_discrete_map={0: "#10b981", 1: "#ef4444"}
            )
            fig_violin.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"}
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    
    with tabs[1]:
        # Risk factor analysis
        risk_thresholds = {
            'Glucose': 140,
            'BloodPressure': 90,
            'BMI': 30,
            'Age': 45
        }
        
        risk_analysis = []
        for feature, threshold in risk_thresholds.items():
            if feature in df.columns:
                high_risk = df[df[feature] > threshold]
                risk_rate = high_risk['Outcome'].mean() * 100
                prevalence = len(high_risk) / len(df) * 100
                
                risk_analysis.append({
                    'Risk Factor': f"{feature} > {threshold}",
                    'Diabetes Rate (%)': risk_rate,
                    'Prevalence (%)': prevalence,
                    'Risk Ratio': risk_rate / (df['Outcome'].mean() * 100)
                })
        
        if risk_analysis:
            risk_df = pd.DataFrame(risk_analysis)
            st.dataframe(risk_df.round(2), use_container_width=True)
            
            # Visualize risk factors
            fig_risk = px.scatter(
                risk_df,
                x='Prevalence (%)',
                y='Diabetes Rate (%)',
                size='Risk Ratio',
                text='Risk Factor',
                title="Risk Factor Analysis: Prevalence vs Diabetes Rate",
                color='Risk Ratio',
                color_continuous_scale='reds'
            )
            fig_risk.update_traces(textposition="top center")
            fig_risk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"}
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tabs[2]:
        # Trend analysis by age groups
        df_age_groups = df.copy()
        df_age_groups['Age_Group'] = pd.cut(
            df_age_groups['Age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['<30', '30-40', '40-50', '50-60', '60+']
        )
        
        age_analysis = df_age_groups.groupby('Age_Group').agg({
            'Outcome': ['count', 'sum', 'mean'],
            'Glucose': 'mean',
            'BMI': 'mean',
            'BloodPressure': 'mean'
        }).round(2)
        
        # Flatten column names
        age_analysis.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in age_analysis.columns]
        age_analysis = age_analysis.reset_index()
        
        # Plot trends
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Diabetes Rate by Age', 'Average Glucose by Age', 
                           'Average BMI by Age', 'Average BP by Age'],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Add traces
        metrics = [
            ('mean_Outcome', 'Diabetes Rate'),
            ('mean_Glucose', 'Glucose'),
            ('mean_BMI', 'BMI'),
            ('mean_BloodPressure', 'Blood Pressure')
        ]
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (metric, title) in enumerate(metrics):
            row, col = positions[i]
            fig_trends.add_scatter(
                x=age_analysis['Age_Group'],
                y=age_analysis[metric],
                mode='lines+markers',
                name=title,
                row=row, col=col,
                showlegend=False
            )
        
        fig_trends.update_layout(
            height=600,
            title_text="Health Metrics Trends by Age Group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig_trends, use_container_width=True)

# Enhanced model insights page
def show_enhanced_model_insights_page(models, best_model_name, df):
    """Enhanced model insights with comprehensive evaluation"""
    st.markdown('<div class="section-header">🔍 Advanced Model Analysis</div>', unsafe_allow_html=True)
    
    # Model performance overview
    st.markdown("### 🎯 Model Performance Overview")
    
    # Create performance comparison
    performance_data = []
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    for name, model_info in models.items():
        model = model_info['model']
        
        # Get predictions
        if model_info['needs_scaling']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            test_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        auc_score = roc_auc_score(y_test, test_prob)
        
        performance_data.append({
            'Model': name,
            'Training Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'AUC Score': auc_score,
            'Overfitting': train_acc - test_acc,
            'Best Model': name == best_model_name
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    best_model_data = perf_df[perf_df['Best Model']]
    
    with col1:
        st.metric(
            "Best Model",
            best_model_name,
            f"{best_model_data['Test Accuracy'].iloc[0]:.1%}"
        )
    
    with col2:
        st.metric(
            "AUC Score",
            f"{best_model_data['AUC Score'].iloc[0]:.3f}",
            "Higher is better"
        )
    
    with col3:
        overfitting = best_model_data['Overfitting'].iloc[0]
        overfitting_status = "Good" if overfitting < 0.05 else "Moderate" if overfitting < 0.1 else "High"
        st.metric(
            "Overfitting",
            f"{overfitting:.3f}",
            overfitting_status
        )
    
    # Model comparison chart
    st.markdown("### 📊 Model Comparison")
    
    fig_comparison = px.scatter(
        perf_df,
        x='Test Accuracy',
        y='AUC Score',
        size='Training Accuracy',
        color='Model',
        title="Model Performance Comparison",
        hover_data=['Overfitting']
    )
    
    # Highlight best model
    best_model_row = perf_df[perf_df['Best Model']]
    fig_comparison.add_scatter(
        x=best_model_row['Test Accuracy'],
        y=best_model_row['AUC Score'],
        mode='markers',
        marker=dict(size=20, color='gold', symbol='star'),
        name='Best Model',
        showlegend=True
    )
    
    fig_comparison.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"}
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed performance table
    st.markdown("### 📋 Detailed Performance Metrics")
    
    display_df = perf_df.drop('Best Model', axis=1).round(4)
    st.dataframe(display_df, use_container_width=True)
    
    # Model-specific analysis
    st.markdown("### 🔬 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select model for detailed analysis:",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name)
    )
    
    model_info = models[selected_model]
    model = model_info['model']
    
    # Get predictions for selected model
    if model_info['needs_scaling']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        test_pred = model.predict(X_test_scaled)
        test_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, test_pred)
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Diabetes', 'Diabetes'],
            y=['No Diabetes', 'Diabetes'],
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification metrics
        st.markdown("#### Classification Report")
        report = classification_report(y_test, test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
    
    with col2:
        # ROC Curve
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        auc = roc_auc_score(y_test, test_prob)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{selected_model} (AUC = {auc:.3f})',
            line=dict(color='#6366f1', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Prediction distribution
        st.markdown("#### Prediction Distribution")
        prob_df = pd.DataFrame({
            'Probability': test_prob,
            'Actual': y_test.values
        })
        
        fig_dist = px.histogram(
            prob_df,
            x='Probability',
            color='Actual',
            nbins=20,
            title='Prediction Probability Distribution',
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            barmode='overlay',
            opacity=0.7
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        st.markdown("### 🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"{selected_model} Feature Importance",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Cross-validation results
    st.markdown("### 🔄 Cross-Validation Results")
    with st.spinner("Running cross-validation (this may take a moment)..."):
        if model_info['needs_scaling']:
            X_scaled = scaler.fit_transform(X)
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        cv_df = pd.DataFrame({
            'Fold': range(1, 6),
            'Accuracy': cv_scores
        })
        
        st.write(f"Average CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        fig_cv = px.line(
            cv_df,
            x='Fold',
            y='Accuracy',
            markers=True,
            title="Cross-Validation Performance",
            range_y=[0, 1]
        )
        fig_cv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Inter"}
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# Enhanced about page
def show_enhanced_about_page():
    """Enhanced about page with detailed information"""
    st.markdown('<div class="section-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Advanced Diabetes Prediction System</h3>
        <p>This application uses machine learning to assess diabetes risk based on key health indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Comprehensive Analysis</h4>
            <p>• Multiple visualization tools for data exploration</p>
            <p>• Detailed statistical analysis of health metrics</p>
            <p>• Feature importance and correlation analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Advanced ML Models</h4>
            <p>• Ensemble of multiple machine learning algorithms</p>
            <p>• Model performance comparison</p>
            <p>• Detailed evaluation metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Risk Assessment</h4>
            <p>• Personalized diabetes risk prediction</p>
            <p>• Confidence scoring for predictions</p>
            <p>• Risk progression timeline</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Actionable Insights</h4>
            <p>• Personalized health recommendations</p>
            <p>• Risk factor identification</p>
            <p>• Preventive care suggestions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Dataset Information")
    st.markdown("""
    The Pima Indians Diabetes Dataset contains health metrics from females of Pima Indian heritage.
    It includes diagnostic measurements that can be used to predict diabetes onset.
    
    **Features:**
    - Pregnancies: Number of times pregnant
    - Glucose: Plasma glucose concentration
    - BloodPressure: Diastolic blood pressure (mm Hg)
    - SkinThickness: Triceps skin fold thickness (mm)
    - Insulin: 2-Hour serum insulin (mu U/ml)
    - BMI: Body mass index (weight in kg/(height in m)²)
    - DiabetesPedigreeFunction: Diabetes likelihood function
    - Age: Age in years
    - Outcome: Class variable (0 or 1)
    """)
    
    st.markdown("### 🛠️ Technical Details")
    st.markdown("""
    **Machine Learning Models:**
    - Random Forest Classifier
    - Logistic Regression
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    
    **Evaluation Metrics:**
    - Accuracy
    - AUC-ROC Score
    - Confusion Matrix
    - Classification Report
    
    **Technologies Used:**
    - Python 3
    - Streamlit
    - Scikit-learn
    - Pandas
    - NumPy
    - Plotly
    """)
    
    st.markdown("### 📜 License")
    st.markdown("""
    This project is open-source and available under the MIT License.
    """)

def show_home_page():
    """Show the home/welcome page"""
    st.markdown("""
    <div class="feature-card fade-in">
        <h2>Welcome to the Advanced Diabetes Prediction System</h2>
        <p>This application helps healthcare professionals and individuals assess diabetes risk using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>🚀 Getting Started</h3>
            <ol>
                <li>Upload your diabetes dataset in the sidebar</li>
                <li>Navigate to different sections using the menu</li>
                <li>Explore data, run predictions, and analyze results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>📊 Sample Data Statistics</h3>
            <p>Typical diabetes datasets contain:</p>
            <ul>
                <li>768 patients (average sample size)</li>
                <li>8 key health indicators</li>
                <li>Approximately 35% diabetic cases</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>🔍 Key Health Indicators</h3>
            <p>The model analyzes these critical factors:</p>
            <ul>
                <li>Glucose levels</li>
                <li>Blood pressure</li>
                <li>BMI (Body Mass Index)</li>
                <li>Insulin levels</li>
                <li>Age and genetic factors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>📈 Why Use This Tool?</h3>
            <ul>
                <li>Early diabetes risk detection</li>
                <li>Data-driven health insights</li>
                <li>Personalized risk assessment</li>
                <li>Comprehensive analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card fade-in">
        <h3>📝 Next Steps</h3>
        <p>To begin, upload your dataset using the sidebar. If you don't have one, you can download a sample dataset below.</p>
        <p>
            <a href="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database" target="_blank" class="stButton">
                Download Sample Dataset
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main application function
def main():
    """Main application function"""
    # Inject CSS
    inject_css()
    
    # App title
    st.markdown('<h1 class="main-header">Advanced Diabetes Prediction System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    
    # Sidebar with file upload
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "🎯 Risk Assessment", "📊 Data Analysis", "🔍 Model Insights", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("## 📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload your diabetes dataset (CSV)",
            type=["csv"],
            help="Dataset should include standard diabetes features"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading and validating data..."):
                st.session_state.df = load_data(uploaded_file)
                
                # Train models if not already trained
                if st.session_state.models is None:
                    with st.spinner("Training machine learning models..."):
                        st.session_state.models, st.session_state.best_model, st.session_state.scaler = train_models(st.session_state.df)
        
        st.markdown("---")
        st.markdown("## ℹ️ Information")
        st.markdown("""
        This application uses machine learning to predict diabetes risk based on health indicators.
        
        **Sample Data Format:**
        ```
        Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
        6,148,72,35,0,33.6,0.627,50,1
        1,85,66,29,0,26.6,0.351,31,0
        ```
        """)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Risk Assessment" and st.session_state.df is not None:
        show_enhanced_prediction_page(
            st.session_state.models,
            st.session_state.scaler,
            st.session_state.df,
            st.session_state.best_model
        )
    elif page == "📊 Data Analysis" and st.session_state.df is not None:
        stats = get_comprehensive_stats(st.session_state.df)
        feature_analysis = get_comprehensive_feature_analysis(st.session_state.df, st.session_state.models)
        show_enhanced_data_analysis_page(st.session_state.df, stats, feature_analysis)
    elif page == "🔍 Model Insights" and st.session_state.models is not None:
        show_enhanced_model_insights_page(
            st.session_state.models,
            st.session_state.best_model,
            st.session_state.df
        )
    elif page == "ℹ️ About":
        show_enhanced_about_page()
    else:
        if page != "🏠 Home":
            st.warning("Please upload a dataset to access this page")
        show_home_page()

if __name__ == "__main__":
    main()
