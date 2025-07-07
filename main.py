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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-positive {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .prediction-negative {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    """Load the diabetes dataset"""
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except FileNotFoundError:
        st.error("❌ diabetes.csv file not found. Please ensure the file is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Train and cache model
@st.cache_resource
def load_model():
    """Load or train the diabetes prediction model"""
    try:
        # Try to load a pre-trained model
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # If no model exists, train a new one
        df = load_data()
        
        # Prepare features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train, y_train)
        
        # Save the model for next time
        try:
            with open('diabetes_model.pkl', 'wb') as f:
                pickle.dump(model, f)
        except:
            pass  # If we can't save, that's okay
        
        return model

# Feature importance analysis
@st.cache_data
def get_feature_importance():
    """Get feature importance from the model"""
    model = load_model()
    df = load_data()
    feature_names = df.drop('Outcome', axis=1).columns
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# Data statistics
@st.cache_data
def get_data_stats():
    """Get basic statistics about the dataset"""
    df = load_data()
    total_patients = len(df)
    diabetic_patients = df['Outcome'].sum()
    non_diabetic_patients = total_patients - diabetic_patients
    
    return {
        'total': total_patients,
        'diabetic': diabetic_patients,
        'non_diabetic': non_diabetic_patients,
        'diabetic_percentage': (diabetic_patients / total_patients) * 100
    }

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Prediction App</h1>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model = load_model()
    stats = get_data_stats()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🎯 Prediction", "📊 Data Analysis", "🔍 Model Insights", "ℹ️ About"]
    )
    
    if page == "🎯 Prediction":
        show_prediction_page(model, df)
    elif page == "📊 Data Analysis":
        show_data_analysis_page(df, stats)
    elif page == "🔍 Model Insights":
        show_model_insights_page(model, df)
    elif page == "ℹ️ About":
        show_about_page()

def show_prediction_page(model, df):
    """Show the main prediction interface"""
    st.header("🎯 Diabetes Risk Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Patient Information")
        
        # Input features
        pregnancies = st.slider('Number of Pregnancies', 0, 17, 1, help="Number of times pregnant")
        glucose = st.slider('Glucose Level (mg/dL)', 0, 200, 120, help="Plasma glucose concentration")
        blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 122, 70, help="Diastolic blood pressure")
        skin_thickness = st.slider('Skin Thickness (mm)', 0, 99, 20, help="Triceps skin fold thickness")
        insulin = st.slider('Insulin Level (mu U/ml)', 0, 846, 79, help="2-Hour serum insulin")
        bmi = st.slider('BMI (kg/m²)', 0.0, 67.1, 25.0, help="Body mass index")
        dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, help="Diabetes likelihood based on family history")
        age = st.slider('Age (years)', 21, 81, 30, help="Age in years")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        # Prediction button
        if st.button("🔍 Predict Diabetes Risk", type="primary"):
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Store results in session state
            st.session_state.prediction = prediction
            st.session_state.prediction_proba = prediction_proba
            st.session_state.input_data = input_data
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            prediction_proba = st.session_state.prediction_proba
            input_data = st.session_state.input_data
            
            # Display prediction
            if prediction == 1:
                st.markdown(f'''
                <div class="prediction-positive">
                    <h3>⚠️ High Risk of Diabetes</h3>
                    <p>The model predicts a <strong>{prediction_proba[1]*100:.1f}%</strong> probability of diabetes.</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Risk recommendations
                st.subheader("🎯 Recommendations")
                st.write("""
                - 🏥 **Consult a healthcare provider** for proper diagnosis and treatment
                - 🥗 **Monitor diet**: Reduce sugar and refined carbs
                - 🏃 **Increase physical activity**: At least 150 minutes per week
                - ⚖️ **Maintain healthy weight**: If BMI is high
                - 🩺 **Regular health check-ups**: Monitor blood glucose levels
                """)
            else:
                st.markdown(f'''
                <div class="prediction-negative">
                    <h3>✅ Low Risk of Diabetes</h3>
                    <p>The model predicts a <strong>{prediction_proba[0]*100:.1f}%</strong> probability of no diabetes.</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Prevention recommendations
                st.subheader("🛡️ Prevention Tips")
                st.write("""
                - 🥗 **Maintain healthy diet**: Continue balanced nutrition
                - 🏃 **Stay active**: Regular exercise is key
                - ⚖️ **Monitor weight**: Keep BMI in healthy range
                - 🩺 **Regular check-ups**: Annual health screenings
                - 🚭 **Avoid smoking**: Reduces diabetes risk
                """)
            
            # Probability visualization
            st.subheader("📈 Probability Breakdown")
            
            prob_df = pd.DataFrame({
                'Outcome': ['No Diabetes', 'Diabetes'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            
            fig = px.bar(
                prob_df, 
                x='Outcome', 
                y='Probability',
                color='Outcome',
                color_discrete_map={'No Diabetes': '#2e7d32', 'Diabetes': '#c62828'},
                title="Prediction Probabilities"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display input summary
            st.subheader("📋 Input Summary")
            st.dataframe(input_data.T, use_container_width=True)
            
        else:
            st.info("👆 Please enter patient information and click 'Predict' to see results.")

def show_data_analysis_page(df, stats):
    """Show data analysis and visualizations"""
    st.header("📊 Dataset Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=stats['total']
        )
    
    with col2:
        st.metric(
            label="Diabetic Cases",
            value=stats['diabetic']
        )
    
    with col3:
        st.metric(
            label="Non-Diabetic Cases",
            value=stats['non_diabetic']
        )
    
    with col4:
        st.metric(
            label="Diabetes Rate",
            value=f"{stats['diabetic_percentage']:.1f}%"
        )
    
    # Outcome distribution
    st.subheader("🎯 Outcome Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=[stats['non_diabetic'], stats['diabetic']],
            names=['No Diabetes', 'Diabetes'],
            title="Diabetes Distribution",
            color_discrete_map={'No Diabetes': '#2e7d32', 'Diabetes': '#c62828'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        outcome_counts = df['Outcome'].value_counts()
        fig_bar = px.bar(
            x=['No Diabetes', 'Diabetes'],
            y=[outcome_counts[0], outcome_counts[1]],
            color=['No Diabetes', 'Diabetes'],
            color_discrete_map={'No Diabetes': '#2e7d32', 'Diabetes': '#c62828'},
            title="Case Counts"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    
    # Select features to visualize
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('Outcome')
    
    selected_features = st.multiselect(
        "Select features to visualize:",
        numeric_columns,
        default=numeric_columns[:4]
    )
    
    if selected_features:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_features[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, feature in enumerate(selected_features[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Histogram for each outcome
            for outcome in [0, 1]:
                data = df[df['Outcome'] == outcome][feature]
                fig.add_histogram(
                    x=data,
                    name=f"{'Diabetes' if outcome == 1 else 'No Diabetes'}",
                    row=row, col=col,
                    showlegend=(i == 0),
                    opacity=0.7,
                    marker_color='#c62828' if outcome == 1 else '#2e7d32'
                )
        
        fig.update_layout(height=600, title_text="Feature Distributions by Outcome")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlations")
    
    corr_matrix = df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

def show_model_insights_page(model, df):
    """Show model performance and insights"""
    st.header("🔍 Model Performance & Insights")
    
    # Feature importance
    st.subheader("⭐ Feature Importance")
    
    importance_df = get_feature_importance()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Ranking",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        # Table
        st.subheader("📋 Importance Values")
        importance_df['Importance'] = importance_df['Importance'].round(4)
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    # Model performance metrics
    st.subheader("📊 Model Performance")
    
    # Get model predictions on full dataset for demonstration
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        st.metric("Test Samples", len(y_test))
    
    with col3:
        st.metric("Features Used", len(X.columns))
    
    # Confusion matrix
    st.subheader("🎯 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=['No Diabetes', 'Diabetes'],
        y=['No Diabetes', 'Diabetes'],
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig_cm.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
            )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification report
    st.subheader("📈 Classification Report")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

def show_about_page():
    """Show information about the app and dataset"""
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ## 🩺 Diabetes Prediction App
    
    This application uses machine learning to predict the likelihood of diabetes based on various health metrics.
    
    ### 📊 Dataset Information
    - **Source**: Pima Indians Diabetes Dataset
    - **Total Records**: 768 patients
    - **Features**: 8 health-related measurements
    - **Target**: Diabetes outcome (0 = No Diabetes, 1 = Diabetes)
    
    ### 🔬 Features Used
    1. **Pregnancies**: Number of times pregnant
    2. **Glucose**: Plasma glucose concentration (mg/dL)
    3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
    4. **Skin Thickness**: Triceps skin fold thickness (mm)
    5. **Insulin**: 2-Hour serum insulin (mu U/ml)
    6. **BMI**: Body mass index (kg/m²)
    7. **Diabetes Pedigree Function**: Likelihood based on family history
    8. **Age**: Age in years
    
    ### 🤖 Machine Learning Model
    - **Algorithm**: Random Forest Classifier
    - **Training Method**: Supervised learning
    - **Validation**: Train-test split (80/20)
    - **Performance**: Accuracy metrics available in Model Insights
    
    ### ⚠️ Important Disclaimers
    - This app is for **educational purposes only**
    - **Not a substitute** for professional medical advice
    - Always consult healthcare providers for medical decisions
    - Results should not be used for self-diagnosis
    
    ### 🔧 Technical Details
    - Built with Streamlit and scikit-learn
    - Interactive visualizations using Plotly
    - Responsive design for various screen sizes
    - Cached models for improved performance
    
    ### 👨‍💻 Usage Tips
    1. Use the **Prediction** page for individual risk assessment
    2. Explore the **Data Analysis** page to understand the dataset
    3. Check **Model Insights** for performance metrics
    4. Adjust input parameters to see how they affect predictions
    
    ### 📚 Learn More
    - [Diabetes Information - CDC](https://www.cdc.gov/diabetes/)
    - [Machine Learning in Healthcare](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6616181/)
    - [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
    """)

if __name__ == "__main__":
    main()
