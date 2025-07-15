with source as (
    select * from {{ source('university', 'course_grades') }}
),

renamed as (
    select
        student_id,
        course_code,
        grade,
        CURRENT_TIMESTAMP as loaded_at
    from source
)

select * from renamed