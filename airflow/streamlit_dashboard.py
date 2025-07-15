import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configuration
DB_PATH = "data/university.db"
MODEL_PATH = "data/processed/risk_model.pkl"
SCALER_PATH = "data/processed/scaler.pkl"

# Set page config
st.set_page_config(
    page_title="University Student Success Analytics",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .st-bq {
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffa500;
        font-weight: bold;
    }
    .low-risk {
        color: #0f9d58;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    students_df = pd.read_sql("SELECT * FROM students", conn)
    courses_df = pd.read_sql("SELECT * FROM course_grades", conn)
    conn.close()
    return students_df, courses_df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_risk(student_data, model, scaler):
    # Prepare features in same order as training
    features = [
        'gpa', 'attendance_rate', 'participation_rate',
        'financial_aid', 'first_generation', 'international_student',
        'current_year', 'gender_code', 'major_code'
    ]
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([student_data])[features]
    
    # Scale features
    scaled_input = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    return prediction[0], prediction_proba[0][1]

def main():
    st.title("🎓 University Student Success Analytics Platform")
    st.markdown("""
        This dashboard provides insights into student performance and identifies at-risk students 
        for early intervention.
    """)
    
    # Load data and model
    students_df, courses_df = load_data()
    model, scaler = load_model()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_major = st.sidebar.selectbox(
        "Select Major", 
        ['All'] + sorted(students_df['major'].unique().tolist())
    )
    selected_year = st.sidebar.selectbox(
        "Select Enrollment Year", 
        ['All'] + sorted(students_df['enrollment_year'].unique().tolist())
    )
    risk_filter = st.sidebar.multiselect(
        "Select Risk Categories",
        ['Low', 'Medium', 'High', 'Critical'],
        default=['High', 'Critical']
    )
    
    # Apply filters
    filtered_df = students_df.copy()
    if selected_major != 'All':
        filtered_df = filtered_df[filtered_df['major'] == selected_major]
    if selected_year != 'All':
        filtered_df = filtered_df[filtered_df['enrollment_year'] == selected_year]
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_category'].isin(risk_filter)]
    
    # Overview metrics
    st.header("📊 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Students</h3>
                <h2>{:,}</h2>
            </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    with col2:
        avg_gpa = filtered_df['gpa'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3>Average GPA</h3>
                <h2>{avg_gpa:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_counts = filtered_df['risk_category'].value_counts(normalize=True) * 100
        high_critical = risk_counts.get('High', 0) + risk_counts.get('Critical', 0)
        st.markdown(f"""
            <div class="metric-card">
                <h3>At Risk Students</h3>
                <h2>{high_critical:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_attendance = filtered_df['attendance_rate'].mean() * 100
        st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Attendance</h3>
                <h2>{avg_attendance:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trends & Distributions", 
        "🎓 Department Analysis", 
        "⚠️ Risk Analysis",
        "🔍 Student Explorer"
    ])
    
    with tab1:
        st.subheader("Academic Trends and Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # GPA Distribution
            fig = px.histogram(
                filtered_df, 
                x='gpa', 
                nbins=20,
                title='GPA Distribution',
                labels={'gpa': 'GPA'},
                color_discrete_sequence=['#4285f4']
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Risk Category Distribution
            risk_counts = filtered_df['risk_category'].value_counts().reset_index()
            risk_counts.columns = ['Risk Category', 'Count']
            
            fig = px.pie(
                risk_counts,
                names='Risk Category',
                values='Count',
                title='Risk Category Distribution',
                color='Risk Category',
                color_discrete_map={
                    'Low': '#0f9d58',
                    'Medium': '#ffa500',
                    'High': '#ff6b4b',
                    'Critical': '#db4437'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # GPA Trend by Enrollment Year
        st.subheader("GPA Trend by Enrollment Year")
        gpa_trend = filtered_df.groupby('enrollment_year')['gpa'].mean().reset_index()
        
        fig = px.line(
            gpa_trend,
            x='enrollment_year',
            y='gpa',
            title='Average GPA by Enrollment Year',
            markers=True,
            labels={'enrollment_year': 'Enrollment Year', 'gpa': 'Average GPA'}
        )
        fig.update_traces(line_color='#4285f4', line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Department-Level Analysis")
        
        # Department metrics
        dept_metrics = filtered_df.groupby('major').agg({
            'student_id': 'count',
            'gpa': 'mean',
            'attendance_rate': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        dept_metrics.columns = ['Major', 'Student Count', 'Average GPA', 'Average Attendance', 'Average Risk Score']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department GPA Comparison
            fig = px.bar(
                dept_metrics.sort_values('Average GPA', ascending=False),
                x='Major',
                y='Average GPA',
                title='Average GPA by Major',
                color='Average GPA',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Department Risk Comparison
            fig = px.bar(
                dept_metrics.sort_values('Average Risk Score', ascending=False),
                x='Major',
                y='Average Risk Score',
                title='Average Risk Score by Major',
                color='Average Risk Score',
                color_continuous_scale='Inferno'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show department metrics table
        st.subheader("Department Metrics")
        st.dataframe(
            dept_metrics.sort_values('Average Risk Score', ascending=False),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Student Risk Analysis")
        
        # Risk factors correlation
        st.subheader("Risk Factors Correlation")
        
        corr_df = filtered_df[[
            'gpa', 'attendance_rate', 'participation_rate',
            'financial_aid', 'first_generation', 'international_student',
            'risk_score'
        ]]
        corr_df['financial_aid'] = corr_df['financial_aid'].astype(int)
        corr_df['first_generation'] = corr_df['first_generation'].astype(int)
        corr_df['international_student'] = corr_df['international_student'].astype(int)
        
        corr_matrix = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            ax=ax,
            fmt=".2f",
            annot_kws={"size": 10}
        )
        ax.set_title("Correlation Between Risk Factors", pad=20)
        st.pyplot(fig)
        
        # High risk students table
        st.subheader("High Risk Students")
        high_risk_df = filtered_df[filtered_df['risk_category'].isin(['High', 'Critical'])]
        st.dataframe(
            high_risk_df.sort_values('risk_score', ascending=False)[[
                'student_id', 'first_name', 'last_name', 'major', 'gpa', 
                'attendance_rate', 'risk_score', 'risk_category'
            ]],
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Individual Student Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Student selector
            student_id = st.selectbox(
                "Select Student",
                filtered_df['student_id'].unique()
            )
            
            # Get student data
            student_data = filtered_df[filtered_df['student_id'] == student_id].iloc[0]
            student_courses = courses_df[courses_df['student_id'] == student_id]
            
            # Display student info
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Student Information</h3>
                    <p><strong>Name:</strong> {student_data['first_name']} {student_data['last_name']}</p>
                    <p><strong>Major:</strong> {student_data['major']}</p>
                    <p><strong>Enrollment Year:</strong> {student_data['enrollment_year']}</p>
                    <p><strong>Current Year:</strong> {student_data['current_year']}</p>
                    <p><strong>GPA:</strong> {student_data['gpa']:.2f}</p>
                    <p><strong>Attendance:</strong> {student_data['attendance_rate']*100:.1f}%</p>
                    <p><strong>Participation:</strong> {student_data['participation_rate']*100:.1f}%</p>
                    <p><strong>Risk Category:</strong> <span class="{student_data['risk_category'].lower()}-risk">
                        {student_data['risk_category']}
                    </span></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Predict risk with current model
            if st.button("Re-evaluate Risk"):
                prediction, proba = predict_risk(student_data, model, scaler)
                risk_label = "At Risk" if prediction == 1 else "Not At Risk"
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Model Prediction</h3>
                        <p><strong>Prediction:</strong> {risk_label}</p>
                        <p><strong>Probability:</strong> {proba*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Course performance
            st.markdown("""
                <div class="metric-card">
                    <h3>Course Performance</h3>
            """, unsafe_allow_html=True)
            
            if not student_courses.empty:
                fig = px.bar(
                    student_courses,
                    x='course_code',
                    y='grade',
                    title='Course Grades',
                    labels={'course_code': 'Course', 'grade': 'Grade'},
                    color='grade',
                    color_continuous_scale='Viridis',
                    range_y=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No course data available for this student")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Student performance over time (simulated)
        st.subheader("Performance Trend")
        performance_trend = pd.DataFrame({
            'semester': ['Fall 2021', 'Spring 2022', 'Fall 2022', 'Spring 2023'],
            'gpa': [
                max(0, min(4.0, student_data['gpa'] + np.random.normal(0, 0.2))),
                max(0, min(4.0, student_data['gpa'] + np.random.normal(0, 0.2))),
                max(0, min(4.0, student_data['gpa'] + np.random.normal(0, 0.2))),
                max(0, min(4.0, student_data['gpa'] + np.random.normal(0, 0.2)))
            ],
            'attendance': [
                max(0, min(1, student_data['attendance_rate'] + np.random.normal(0, 0.05))) * 100,
                max(0, min(1, student_data['attendance_rate'] + np.random.normal(0, 0.05))) * 100,
                max(0, min(1, student_data['attendance_rate'] + np.random.normal(0, 0.05))) * 100,
                max(0, min(1, student_data['attendance_rate'] + np.random.normal(0, 0.05))) * 100
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=performance_trend['semester'],
            y=performance_trend['gpa'],
            name='GPA',
            line=dict(color='#4285f4', width=3),
            yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=performance_trend['semester'],
            y=performance_trend['attendance'],
            name='Attendance (%)',
            line=dict(color='#0f9d58', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Academic Performance Over Time',
            yaxis=dict(
                title='GPA',
                range=[0, 4],
                title_font=dict(color='#4285f4'),  # Changed from titlefont to title_font
                tickfont=dict(color='#4285f4')
            ),
            yaxis2=dict(
                title='Attendance (%)',
                range=[0, 100],
                overlaying='y',
                side='right',
                title_font=dict(color='#0f9d58'),  # Changed from titlefont to title_font
                tickfont=dict(color='#0f9d58')
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()