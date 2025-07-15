import os
import sqlite3
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import sqlalchemy
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "student_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_student_data.csv")
DB_PATH = os.path.join(DATA_DIR, "university.db")
MODEL_PATH = os.path.join(DATA_DIR, "processed", "risk_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "processed", "scaler.pkl")

# Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

class StudentDataGenerator:
    def __init__(self, num_students=1000):
        self.fake = Faker()
        self.num_students = num_students
        self.majors = [
            'Computer Science', 'Engineering', 'Mathematics', 'Physics', 
            'Biology', 'Chemistry', 'Psychology', 'Economics', 
            'Business', 'English', 'History', 'Art'
        ]
        self.courses = {
            'Computer Science': ['CS101', 'CS201', 'CS301', 'CS401', 'MATH101'],
            'Engineering': ['ENG101', 'ENG201', 'ENG301', 'MATH101', 'PHYS101'],
            'Mathematics': ['MATH101', 'MATH201', 'MATH301', 'MATH401', 'STAT101'],
            'Physics': ['PHYS101', 'PHYS201', 'PHYS301', 'MATH101', 'CHEM101'],
            'Biology': ['BIO101', 'BIO201', 'BIO301', 'CHEM101', 'STAT101'],
            'Chemistry': ['CHEM101', 'CHEM201', 'CHEM301', 'PHYS101', 'MATH101'],
            'Psychology': ['PSY101', 'PSY201', 'PSY301', 'STAT101', 'BIO101'],
            'Economics': ['ECON101', 'ECON201', 'ECON301', 'MATH101', 'STAT101'],
            'Business': ['BUS101', 'BUS201', 'BUS301', 'ECON101', 'MATH101'],
            'English': ['ENG101', 'ENG201', 'ENG301', 'HIST101', 'ART101'],
            'History': ['HIST101', 'HIST201', 'HIST301', 'POL101', 'ENG101'],
            'Art': ['ART101', 'ART201', 'ART301', 'HIST101', 'ENG101']
        }

    def generate_student_data(self):
        students = []
        
        for _ in range(self.num_students):
            major = np.random.choice(self.majors)
            gpa = np.random.normal(loc=3.0, scale=0.5)
            gpa = max(0, min(4.0, gpa))  # Ensure GPA is between 0 and 4.0
            
            # Generate risk factors
            attendance = np.random.beta(5, 2)  # Most students attend regularly
            participation = np.random.beta(4, 3)
            financial_aid = np.random.choice([True, False], p=[0.7, 0.3])
            first_gen = np.random.choice([True, False], p=[0.3, 0.7])
            international = np.random.choice([True, False], p=[0.1, 0.9])
            
            # Generate some course grades
            student_courses = self.courses.get(major, [])
            course_grades = {}
            for course in student_courses:
                base_grade = gpa * 25  # Convert GPA to percentage scale
                variation = np.random.normal(0, 10)
                course_grade = max(0, min(100, base_grade + variation))
                course_grades[course] = course_grade
            
            student = {
                'student_id': self.fake.unique.random_number(digits=8),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'email': self.fake.email(),
                'gender': np.random.choice(['Male', 'Female', 'Other']),
                'date_of_birth': self.fake.date_of_birth(minimum_age=18, maximum_age=25),
                'major': major,
                'enrollment_year': np.random.randint(2018, 2023),
                'gpa': gpa,
                'attendance_rate': attendance,
                'participation_rate': participation,
                'financial_aid': financial_aid,
                'first_generation': first_gen,
                'international_student': international,
                'current_year': np.random.choice([1, 2, 3, 4], p=[0.25, 0.25, 0.25, 0.25]),
                'courses': ', '.join(student_courses),
                **{f'grade_{course}': grade for course, grade in course_grades.items()}
            }
            students.append(student)
        
        df = pd.DataFrame(students)
        return df

    def save_raw_data(self, df):
        df.to_csv(RAW_DATA_PATH, index=False)

class DataProcessor:
    def __init__(self):
        self.engine = create_engine(f'sqlite:///{DB_PATH}')

    def calculate_risk_factors(self, df):
        # Calculate average course grade
        grade_cols = [col for col in df.columns if col.startswith('grade_')]
        df['avg_course_grade'] = df[grade_cols].mean(axis=1)
        
        # Calculate risk score (simple version)
        df['risk_score'] = (
            (4.0 - df['gpa']) * 0.4 + 
            (1 - df['attendance_rate']) * 0.3 + 
            (1 - df['participation_rate']) * 0.2 +
            df['first_generation'].astype(int) * 0.05 +
            df['international_student'].astype(int) * 0.05
        )
        
        # Create risk category
        df['risk_category'] = pd.cut(df['risk_score'], 
                                    bins=[0, 0.3, 0.7, 1.5, 3.0],
                                    labels=['Low', 'Medium', 'High', 'Critical'])
        
        return df

    def process_data(self, raw_df):
        # Clean and transform data
        processed_df = raw_df.copy()
        
        # Calculate additional features
        processed_df = self.calculate_risk_factors(processed_df)
        
        # Convert categorical to numerical
        processed_df['gender_code'] = processed_df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
        processed_df['major_code'] = pd.factorize(processed_df['major'])[0]
        
        # Save processed data
        processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
        
        return processed_df

    def load_to_database(self, df):
        # Create SQLite database and tables
        with sqlite3.connect(DB_PATH) as conn:
            # Students table
            student_cols = [
                'student_id', 'first_name', 'last_name', 'email', 'gender', 
                'date_of_birth', 'major', 'enrollment_year', 'gpa', 
                'attendance_rate', 'participation_rate', 'financial_aid',
                'first_generation', 'international_student', 'current_year',
                'risk_score', 'risk_category'
            ]
            df[student_cols].to_sql('students', conn, if_exists='replace', index=False)
            
            # Courses table
            grade_cols = [col for col in df.columns if col.startswith('grade_')]
            course_data = []
            for _, row in df.iterrows():
                for col in grade_cols:
                    course_name = col.replace('grade_', '')
                    course_data.append({
                        'student_id': row['student_id'],
                        'course_code': course_name,
                        'grade': row[col]
                    })
            pd.DataFrame(course_data).to_sql('course_grades', conn, if_exists='replace', index=False)

class RiskPredictor:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        # Select features for model
        features = [
            'gpa', 'attendance_rate', 'participation_rate',
            'financial_aid', 'first_generation', 'international_student',
            'current_year', 'gender_code', 'major_code'
        ]
        
        X = df[features]
        y = (df['risk_category'].isin(['High', 'Critical'])).astype(int)
        
        return X, y

    def train_model(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        
        return model

    def save_model(self, model):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)

def main():
    print("University Student Success Analytics Platform")
    print("="*50)
    
    # Step 1: Generate synthetic data
    print("\nGenerating synthetic student data...")
    generator = StudentDataGenerator(num_students=1000)
    student_df = generator.generate_student_data()
    generator.save_raw_data(student_df)
    print(f"Generated {len(student_df)} student records")
    
    # Step 2: Process data
    print("\nProcessing data...")
    processor = DataProcessor()
    processed_df = processor.process_data(student_df)
    processor.load_to_database(processed_df)
    print("Data processed and loaded to database")
    
    # Step 3: Train risk prediction model
    print("\nTraining risk prediction model...")
    predictor = RiskPredictor()
    X, y = predictor.prepare_data(processed_df)
    model = predictor.train_model(X, y)
    predictor.save_model(model)
    print("Model trained and saved")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()