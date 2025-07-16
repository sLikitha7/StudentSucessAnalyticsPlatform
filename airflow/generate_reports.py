#!/usr/bin/env python3
"""
Analytics Report Generator for Student Success Platform
Generates CSV reports from the SQLite database
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

# Dynamically resolve the absolute path to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use paths relative to script location (safe for local and Airflow Docker)
DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'university.db')
REPORTS_DIR = os.path.join(SCRIPT_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Saving reports to: {os.path.abspath(REPORTS_DIR)}")

def generate_risk_report():
    """Generate CSV report of at-risk students"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT s.student_id, s.first_name, s.last_name, s.major, s.gpa, 
           s.attendance_rate, s.participation_rate, s.risk_score, s.risk_category
    FROM students s
    WHERE s.risk_category IN ('High', 'Critical')
    ORDER BY s.risk_score DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"at_risk_students_{timestamp}.csv")
    df.to_csv(report_path, index=False)
    print(f"Generated risk report: {report_path}")
    return report_path

def generate_department_report():
    """Generate department-level performance report"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        major as department,
        COUNT(*) as student_count,
        AVG(gpa) as avg_gpa,
        AVG(attendance_rate) as avg_attendance,
        AVG(participation_rate) as avg_participation,
        AVG(risk_score) as avg_risk_score,
        SUM(CASE WHEN risk_category IN ('High', 'Critical') THEN 1 ELSE 0 END) as at_risk_count
    FROM students
    GROUP BY major
    ORDER BY avg_risk_score DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"department_performance_{timestamp}.csv")
    df.to_csv(report_path, index=False)
    print(f"Generated department report: {report_path}")
    return report_path

def generate_course_performance_report():
    """Generate course-level performance report"""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        c.course_code,
        COUNT(*) as student_count,
        AVG(c.grade) as avg_grade,
        SUM(CASE WHEN c.grade < 60 THEN 1 ELSE 0 END) as failing_count,
        AVG(s.gpa) as avg_student_gpa
    FROM course_grades c
    JOIN students s ON c.student_id = s.student_id
    GROUP BY c.course_code
    ORDER BY avg_grade
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"course_performance_{timestamp}.csv")
    df.to_csv(report_path, index=False)
    print(f"Generated course performance report: {report_path}")
    return report_path

def main():
    print("Starting analytics report generation...")
    reports = [
        generate_risk_report(),
        generate_department_report(),
        generate_course_performance_report()
    ]
    print(f"Generated {len(reports)} reports in {os.path.abspath(REPORTS_DIR)}")

if __name__ == "__main__":
    main()
