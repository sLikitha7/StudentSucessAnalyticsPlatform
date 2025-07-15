with students as (
    select * from main_staging."stg_students"
)

select
    student_id,
    first_name,
    last_name,
    email,
    gender,
    date_of_birth,
    major,
    enrollment_year,
    current_year,
    financial_aid,
    first_generation,
    international_student,
    loaded_at
from students