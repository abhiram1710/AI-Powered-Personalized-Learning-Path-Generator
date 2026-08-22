import pandas as pd

# --------------------------------------------------
# Load all required files
# --------------------------------------------------

learning_path = pd.read_csv("personalized_learning_path.csv")
skill_gaps = pd.read_csv("skill_gap_analysis.csv")

print("Personalized learning path loaded!")
print("Skill-gap analysis loaded!")


# --------------------------------------------------
# Student information
# --------------------------------------------------

student_id = "S1"
target_career = (
    "VP / Client, Finance & Strategy Data Scientist Lead – "
    "JPM Private Client"
)

current_profession = "Astronomer"
dominant_intelligence = "Naturalist"
learning_style = "classification and pattern-based learning"
pace = "Fast-paced"
difficulty = "Advanced"
practice_level = "High practice"

learning_recommendation = (
    "Use examples, categorization, comparisons and pattern recognition."
)


# --------------------------------------------------
# Add career information
# --------------------------------------------------

learning_path["target_career"] = target_career


# --------------------------------------------------
# Add learning priority
# --------------------------------------------------

def get_priority(status, stage):

    if status == "Skill Gap - No Course":
        return "High"

    if stage in ["Programming", "Data", "Statistics"]:
        return "High"

    if stage == "Machine Learning":
        return "High"

    return "Medium"


learning_path["priority"] = learning_path.apply(
    lambda row: get_priority(
        row["status"],
        row["stage"]
    ),
    axis=1
)


# --------------------------------------------------
# Add practice recommendation
# --------------------------------------------------

learning_path["practice_recommendation"] = (
    "High-practice exercises with examples, "
    "comparisons and pattern-based problems"
)


# --------------------------------------------------
# Add learning method
# --------------------------------------------------

learning_path["learning_method"] = (
    "Example-based, classification-based and "
    "pattern-recognition learning"
)


# --------------------------------------------------
# Reorder important columns
# --------------------------------------------------

columns = [
    "student_id",
    "target_career",
    "sequence",
    "stage",
    "skill",
    "course_id",
    "course_name",
    "institution",
    "course_score",
    "status",
    "priority",
    "learning_method",
    "practice_recommendation"
]

final_path = learning_path[columns]


# --------------------------------------------------
# Save final learning path
# --------------------------------------------------

output_file = "final_personalized_learning_path.csv"

final_path.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Display final learning path
# --------------------------------------------------

print("\n" + "=" * 80)
print("FINAL PERSONALIZED LEARNING PATH")
print("=" * 80)

print("Student:", student_id)
print("Target Career:", target_career)
print("Current Profession:", current_profession)
print("Dominant Intelligence:", dominant_intelligence)
print("Learning Style:", learning_style)
print("Pace:", pace)
print("Difficulty:", difficulty)
print("Practice Level:", practice_level)

print("\nLearning Recommendation:")
print(learning_recommendation)

print("\n" + "-" * 80)
print("LEARNING PATH")
print("-" * 80)

print(
    final_path[
        [
            "sequence",
            "stage",
            "skill",
            "course_name",
            "status",
            "priority"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

total_skills = len(final_path)

available_courses = (
    final_path["status"] == "Course Available"
).sum()

skill_gaps = (
    final_path["status"] == "Skill Gap - No Course"
).sum()

high_priority = (
    final_path["priority"] == "High"
).sum()


print("\n" + "=" * 80)
print("FINAL PATH GENERATED SUCCESSFULLY!")
print("=" * 80)

print("Total learning steps:", total_skills)
print("Steps with courses:", available_courses)
print("Skill gaps:", skill_gaps)
print("High-priority steps:", high_priority)

print("\nOutput file:")
print(output_file)