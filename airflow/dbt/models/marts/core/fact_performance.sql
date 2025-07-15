with students as (
    select * from {{ ref('stg_students') }}
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