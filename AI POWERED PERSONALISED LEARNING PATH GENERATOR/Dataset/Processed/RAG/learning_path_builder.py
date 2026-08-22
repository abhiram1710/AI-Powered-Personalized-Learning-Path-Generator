import pandas as pd

# --------------------------------------------------
# Load ranked courses
# --------------------------------------------------

ranked_courses = pd.read_csv("ranked_courses.csv")

print("Ranked courses loaded!")
print("Total courses:", len(ranked_courses))


# --------------------------------------------------
# Skill learning order
# --------------------------------------------------

skill_order = {

    # Stage 1
    "Programming": [
        "Python"
    ],

    # Stage 2
    "Data": [
        "Data Integrity",
        "Data Preparation",
        "Data Mining",
        "Data Visualization"
    ],

    # Stage 3
    "Statistics": [
        "Statistical Software",
        "Regressions",
        "Time Series Analysis"
    ],

    # Stage 4
    "Machine Learning": [
        "Machine Learning",
        "Predictive Modeling",
        "Causal-Model Approaches",
        "Text Mining"
    ],

    # Stage 5
    "Databases and Big Data": [
        "Sql",
        "Big Data",
        "Hadoop",
        "Apache Spark",
        "Hive",
        "Pig"
    ],

    # Stage 6
    "Cloud and Tools": [
        "Cloud Platforms",
        "Aws",
        "Tableau"
    ],

    # Stage 7
    "Professional Skills": [
        "Communication",
        "Stakeholder Engagement",
        "Strategic Thinking",
        "Problem Solving",
        "Project Management",
        "Teamwork",
        "Business Acumen",
        "Transparency",
        "Storytelling",
        "Iterative Development"
    ]
}


# --------------------------------------------------
# Create learning path
# --------------------------------------------------

learning_path = []

sequence = 1

for stage, skills in skill_order.items():

    for skill in skills:

        matches = ranked_courses[
            ranked_courses["required_skill"].str.lower()
            == skill.lower()
        ]

        if len(matches) == 0:

            learning_path.append({
                "sequence": sequence,
                "stage": stage,
                "skill": skill,
                "course_id": "",
                "course_name": "",
                "institution": "",
                "course_score": 0,
                "status": "Skill Gap - No Course"
            })

            sequence += 1

        else:

            # Take best ranked course
            best_course = matches.iloc[0]

            learning_path.append({
                "sequence": sequence,
                "stage": stage,
                "skill": skill,
                "course_id": best_course["course_id"],
                "course_name": best_course["course_name"],
                "institution": best_course["institution"],
                "course_score": best_course["course_score"],
                "status": "Course Available"
            })

            sequence += 1


# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

learning_path_df = pd.DataFrame(learning_path)


# --------------------------------------------------
# Save learning path
# --------------------------------------------------

output_file = "learning_path_sequence.csv"

learning_path_df.to_csv(output_file, index=False)


# --------------------------------------------------
# Display learning path
# --------------------------------------------------

print("\n" + "=" * 70)
print("PERSONALISED LEARNING PATH SEQUENCE")
print("=" * 70)

print(
    learning_path_df.to_string(index=False)
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("LEARNING PATH CREATED SUCCESSFULLY!")
print("=" * 70)

print("Total skills:", len(learning_path_df))

print(
    "Courses available:",
    (learning_path_df["status"] == "Course Available").sum()
)

print(
    "Skill gaps:",
    (learning_path_df["status"] == "Skill Gap - No Course").sum()
)

print("Output file:", output_file)