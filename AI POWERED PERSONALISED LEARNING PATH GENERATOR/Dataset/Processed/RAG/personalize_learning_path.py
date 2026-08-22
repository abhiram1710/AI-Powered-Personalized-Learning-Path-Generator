import pandas as pd

# --------------------------------------------------
# Load learning path
# --------------------------------------------------

learning_path = pd.read_csv("learning_path_sequence.csv")

print("Learning path loaded!")
print("Total skills:", len(learning_path))


# --------------------------------------------------
# Student profile - S1
# --------------------------------------------------

student_id = "S1"
current_profession = "Astronomer"
dominant_intelligence = "Naturalist"
average_intelligence_score = 13.62
average_performance = 2.38
performance_category = "Excellent"

learning_style = "classification and pattern-based learning"
learning_recommendation = (
    "Use examples, categorization, comparisons and pattern recognition."
)

pace = "fast"
difficulty = "advanced"
practice_level = "high"


# --------------------------------------------------
# Personalization rules
# --------------------------------------------------

if pace.lower() == "fast":
    recommended_pace = "Fast-paced"

if difficulty.lower() == "advanced":
    recommended_difficulty = "Advanced"

if practice_level.lower() == "high":
    recommended_practice = "High practice"


# --------------------------------------------------
# Add personalization information
# --------------------------------------------------

learning_path["student_id"] = student_id
learning_path["current_profession"] = current_profession
learning_path["dominant_intelligence"] = dominant_intelligence
learning_path["learning_style"] = learning_style
learning_path["learning_recommendation"] = learning_recommendation
learning_path["pace"] = recommended_pace
learning_path["difficulty"] = recommended_difficulty
learning_path["practice_level"] = recommended_practice


# --------------------------------------------------
# Add learning activity recommendation
# --------------------------------------------------

learning_path["activity_recommendation"] = (
    "Use examples, categorization, comparisons and pattern recognition "
    "with high-practice exercises."
)


# --------------------------------------------------
# Save personalized path
# --------------------------------------------------

output_file = "personalized_learning_path.csv"

learning_path.to_csv(output_file, index=False)


# --------------------------------------------------
# Display result
# --------------------------------------------------

print("\n" + "=" * 70)
print("PERSONALIZED LEARNING PATH")
print("=" * 70)

print("Student:", student_id)
print("Current Profession:", current_profession)
print("Dominant Intelligence:", dominant_intelligence)
print("Learning Style:", learning_style)
print("Pace:", recommended_pace)
print("Difficulty:", recommended_difficulty)
print("Practice Level:", recommended_practice)

print("\nLearning Recommendation:")
print(learning_recommendation)

print("\nFirst 10 learning steps:")
print(
    learning_path[
        [
            "sequence",
            "stage",
            "skill",
            "course_name",
            "status"
        ]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("PERSONALIZATION COMPLETED!")
print("=" * 70)

print("Student:", student_id)
print("Total learning steps:", len(learning_path))
print("Output file:", output_file)