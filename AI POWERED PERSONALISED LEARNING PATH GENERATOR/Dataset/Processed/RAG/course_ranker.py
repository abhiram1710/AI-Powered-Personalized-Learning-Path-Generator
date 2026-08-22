import pandas as pd

# --------------------------------------------------
# Load matched courses
# --------------------------------------------------

matched_courses = pd.read_csv("matched_courses.csv")

print("Matched courses loaded!")
print("Total matched records:", len(matched_courses))


# --------------------------------------------------
# Calculate course score
# --------------------------------------------------

# Normalize rating to 0-1
rating_score = matched_courses["average_rating"] / 5

# Log-like review score
review_score = (
    matched_courses["review_count"] /
    matched_courses["review_count"].max()
)

# Final score
matched_courses["course_score"] = (
    0.7 * rating_score +
    0.3 * review_score
)


# --------------------------------------------------
# Rank courses for each skill
# --------------------------------------------------

ranked_courses = (
    matched_courses
    .sort_values(
        ["required_skill", "course_score"],
        ascending=[True, False]
    )
)


# --------------------------------------------------
# Select top 3 courses per skill
# --------------------------------------------------

top_courses = (
    ranked_courses
    .groupby("required_skill")
    .head(3)
    .reset_index(drop=True)
)


# --------------------------------------------------
# Save ranked courses
# --------------------------------------------------

output_file = "ranked_courses.csv"

top_courses.to_csv(output_file, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP COURSES FOR EACH SKILL")
print("=" * 60)

for skill in top_courses["required_skill"].unique():

    print("\n" + "-" * 50)
    print("Skill:", skill)

    skill_courses = top_courses[
        top_courses["required_skill"] == skill
    ]

    print(
        skill_courses[
            [
                "course_id",
                "course_name",
                "institution",
                "average_rating",
                "review_count",
                "course_score"
            ]
        ].to_string(index=False)
    )


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("COURSE RANKING COMPLETED")
print("=" * 60)

print("Total ranked course records:", len(top_courses))
print("Output file:", output_file)