with source as (
    select * from {{ source('university', 'students') }}
),

renamed as (
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
        gpa,
        attendance_rate,
        participation_rate,
        financial_aid,
        first_generation,
        international_student,
        risk_score,
        risk_category,
        CURRENT_TIMESTAMP as loaded_at
    from source
)

select * from renamed