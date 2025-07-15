with performance as (
    select * from {{ ref('fact_performance') }}
),

students as (
    select * from {{ ref('dim_students') }}
),

department_stats as (
    select
        s.major,
        count(*) as student_count,
        avg(p.gpa) as avg_gpa,
        avg(p.attendance_rate) as avg_attendance,
        avg(p.participation_rate) as avg_participation,
        avg(p.risk_score) as avg_risk_score,
        sum(case when p.risk_category in ('High', 'Critical') then 1 else 0 end) as at_risk_count
    from students s
    join performance p on s.student_id = p.student_id
    group by 1
)

select
    major,
    student_count,
    avg_gpa,
    avg_attendance,
    avg_participation,
    avg_risk_score,
    at_risk_count,
    at_risk_count / student_count as at_risk_percentage
from department_stats