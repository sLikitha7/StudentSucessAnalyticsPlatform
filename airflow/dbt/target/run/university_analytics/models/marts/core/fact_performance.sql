
  
    
    
    create  table main_core."fact_performance"
    as
        with students as (
    select * from main_staging."stg_students"
)

select
    student_id,
    gpa,
    attendance_rate,
    participation_rate,
    risk_score,
    risk_category,
    loaded_at
from students

  