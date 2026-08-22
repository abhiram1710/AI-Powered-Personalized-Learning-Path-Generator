import pandas as pd

# --------------------------------------------------
# Load matched courses
# --------------------------------------------------

matched_file = "matched_courses.csv"

matched_courses = pd.read_csv(matched_file)

print("Matched courses loaded!")
print("Total matched records:", len(matched_courses))


# --------------------------------------------------
# Required skills
# --------------------------------------------------

required_skills = [
    "Data Integrity",
    "Communication",
    "Stakeholder Engagement",
    "Strategic Thinking",
    "Problem Solving",
    "Project Management",
    "Teamwork",
    "Business Acumen",
    "Transparency",
    "Python",
    "Sql",
    "Data Visualization",
    "Predictive Modeling",
    "Time Series Analysis",
    "Machine Learning",
    "Data Mining",
    "Tableau",
    "Cloud Platforms",
    "Big Data",
    "Statistical Software",
    "Causal-Model Approaches",
    "Data Preparation",
    "Regressions",
    "Text Mining",
    "Aws",
    "Hadoop",
    "Apache Spark",
    "Hive",
    "Pig",
    "Storytelling",
    "Iterative Development"
]


# --------------------------------------------------
# Find available courses for each skill
# --------------------------------------------------

skill_gap = []

for skill in required_skills:

    matches = matched_courses[
        matched_courses["required_skill"].str.lower()
        == skill.lower()
    ]

    number_of_courses = len(matches)

    if number_of_courses > 0:
        status = "Covered"
    else:
        status = "Gap"

    skill_gap.append({
        "required_skill": skill,
        "matched_courses": number_of_courses,
        "status": status
    })


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

skill_gap_df = pd.DataFrame(skill_gap)


# --------------------------------------------------
# Save result
# --------------------------------------------------

output_file = "skill_gap_analysis.csv"

skill_gap_df.to_csv(output_file, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("SKILL GAP ANALYSIS")
print("=" * 60)

print(skill_gap_df.to_string(index=False))


# --------------------------------------------------
# Summary
# --------------------------------------------------

covered = (skill_gap_df["status"] == "Covered").sum()
gaps = (skill_gap_df["status"] == "Gap").sum()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Total required skills:", len(required_skills))
print("Skills covered:", covered)
print("Skills with gaps:", gaps)

print("\nSkills requiring attention:")

print(
    skill_gap_df[
        skill_gap_df["status"] == "Gap"
    ].to_string(index=False)
)

print("\nSkill-gap analysis saved successfully!")
print("Output file:", output_file)