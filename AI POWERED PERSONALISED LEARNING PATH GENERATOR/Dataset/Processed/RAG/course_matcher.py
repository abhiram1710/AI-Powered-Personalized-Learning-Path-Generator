import pandas as pd
import re


# ============================================================
# LOAD COURSE KNOWLEDGE BASE
# ============================================================

course_file = "..\\Employment skills\\course_knowledge_base.csv"

courses = pd.read_csv(course_file)

print("=" * 70)
print("COURSE KNOWLEDGE BASE LOADED")
print("=" * 70)

print("Total courses:", len(courses))
print("Columns:", list(courses.columns))


# ============================================================
# REQUIRED SKILLS
# ============================================================

required_skills = [
    "Python",
    "Data Integrity",
    "Data Preparation",
    "Data Mining",
    "Data Visualization",
    "Statistical Software",
    "Regressions",
    "Time Series Analysis",
    "Machine Learning",
    "Predictive Modeling",
    "Causal-Model Approaches",
    "Text Mining",
    "Sql",
    "Big Data",
    "Hadoop",
    "Apache Spark",
    "Hive",
    "Pig",
    "Cloud Platforms",
    "Aws",
    "Tableau",
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


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

courses["search_text"] = (
    courses["name"].fillna("").astype(str)
    + " "
    + courses["knowledge"].fillna("").astype(str)
).str.lower()


# ============================================================
# SKILL-SPECIFIC KEYWORDS
# ============================================================

skill_keywords = {

    "Python": [
        "python",
        "programming with python",
        "python programming"
    ],

    "Data Integrity": [
        "data integrity",
        "data quality",
        "data validation",
        "data governance"
    ],

    "Data Preparation": [
        "data preparation",
        "data preprocessing",
        "data cleaning",
        "data wrangling",
        "feature engineering"
    ],

    "Data Mining": [
        "data mining",
        "data mining techniques",
        "knowledge discovery"
    ],

    "Data Visualization": [
        "data visualization",
        "data visualisation",
        "visualization with python",
        "data visualization tools",
        "tableau",
        "visual analytics"
    ],

    "Statistical Software": [
        "statistical software",
        "statistics software",
        "statistical analysis",
        "r programming",
        "spss"
    ],

    "Regressions": [
        "regression",
        "linear regression",
        "logistic regression",
        "regression analysis"
    ],

    "Time Series Analysis": [
        "time series",
        "time-series",
        "time series analysis",
        "forecasting"
    ],

    "Machine Learning": [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "machine learning algorithms"
    ],

    "Predictive Modeling": [
        "predictive modeling",
        "predictive modelling",
        "predictive analytics",
        "prediction models",
        "predictive machine learning"
    ],

    "Causal-Model Approaches": [
        "causal inference",
        "causal modeling",
        "causal modelling",
        "causal analysis",
        "causal machine learning"
    ],

    "Text Mining": [
        "text mining",
        "text analytics",
        "natural language processing",
        "nlp",
        "text analysis"
    ],

    "Sql": [
        "sql",
        "structured query language",
        "sql databases",
        "relational database"
    ],

    "Big Data": [
        "big data",
        "big data analytics",
        "big data processing",
        "big data technologies"
    ],

    "Hadoop": [
        "hadoop",
        "hdfs",
        "mapreduce",
        "hadoop ecosystem"
    ],

    "Apache Spark": [
        "apache spark",
        "pyspark",
        "spark rdd",
        "spark sql",
        "spark streaming",
        "apache spark"
    ],

    "Hive": [
        "apache hive",
        "hiveql",
        "hadoop hive"
    ],

    "Pig": [
        "apache pig",
        "pig latin",
        "apache pig latin",
        "hadoop pig"
    ],

    "Cloud Platforms": [
        "cloud computing",
        "cloud platforms",
        "cloud technology",
        "cloud services",
        "cloud infrastructure",
        "google cloud",
        "microsoft azure",
        "amazon web services"
    ],

    "Aws": [
        "amazon web services",
        "aws cloud",
        "aws",
        "aws services",
        "aws machine learning"
    ],

    "Tableau": [
        "tableau",
        "tableau visualization",
        "tableau data visualization"
    ],

    "Communication": [
        "communication",
        "effective communication",
        "business communication",
        "professional communication"
    ],

    "Stakeholder Engagement": [
        "stakeholder engagement",
        "stakeholder management",
        "stakeholder communication",
        "stakeholder relations"
    ],

    "Strategic Thinking": [
        "strategic thinking",
        "strategic management",
        "business strategy",
        "strategy development"
    ],

    "Problem Solving": [
        "problem solving",
        "problem-solving",
        "creative problem solving",
        "analytical problem solving",
        "computational thinking"
    ],

    "Project Management": [
        "project management",
        "project planning",
        "project management fundamentals",
        "project execution"
    ],

    "Teamwork": [
        "teamwork",
        "team collaboration",
        "collaboration",
        "working in teams"
    ],

    "Business Acumen": [
        "business acumen",
        "business strategy",
        "business analytics",
        "business management"
    ],

    "Transparency": [
        "transparency",
        "data transparency",
        "business ethics",
        "data ethics",
        "ethical decision making"
    ],

    "Storytelling": [
        "storytelling",
        "data storytelling",
        "business storytelling",
        "visual storytelling"
    ],

    "Iterative Development": [
        "iterative development",
        "agile development",
        "agile methodology",
        "iterative design",
        "software development lifecycle"
    ]
}


# ============================================================
# FIND COURSES FOR EACH REQUIRED SKILL
# ============================================================

skill_course_matches = {}


for skill in required_skills:

    keywords = skill_keywords.get(
        skill,
        [skill.lower()]
    )

    # --------------------------------------------------------
    # Create regex pattern
    # --------------------------------------------------------

    pattern = "|".join(
        re.escape(keyword.lower())
        for keyword in keywords
    )

    # --------------------------------------------------------
    # Search courses
    # --------------------------------------------------------

    matches = courses[
        courses["search_text"].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
    ].copy()

    skill_course_matches[skill] = matches

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Skill:", skill)
    print("Matching courses:", len(matches))
    print("=" * 70)

    if len(matches) > 0:

        print(
            matches[
                [
                    "course_id",
                    "name",
                    "institution",
                    "average_rating"
                ]
            ].head(5).to_string(index=False)
        )

    else:

        print("No matching courses found.")


# ============================================================
# COURSE MATCHING SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("COURSE MATCHING SUMMARY")
print("=" * 70)

for skill, matches in skill_course_matches.items():

    print(
        f"{skill}: {len(matches)} courses"
    )


# ============================================================
# CREATE MATCHED COURSE RECORDS
# ============================================================

matched_rows = []


for skill, matches in skill_course_matches.items():

    for _, row in matches.iterrows():

        matched_rows.append({

            "required_skill": skill,

            "course_id": row["course_id"],

            "course_name": row["name"],

            "institution": row["institution"],

            "average_rating": row["average_rating"],

            "review_count": row["review_count"]

        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

matched_courses = pd.DataFrame(
    matched_rows
)


# ============================================================
# SAVE MATCHED COURSES
# ============================================================

output_file = "matched_courses.csv"

matched_courses.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("MATCHED COURSES SAVED SUCCESSFULLY!")
print("=" * 70)

print(
    "Total matched records:",
    len(matched_courses)
)

print(
    "Output file:",
    output_file
)


print("\nFirst 10 matched courses:")

if len(matched_courses) > 0:

    print(
        matched_courses
        .head(10)
        .to_string(index=False)
    )

else:

    print("No courses were matched.")


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 70)
print("COURSE MATCHING COMPLETED!")
print("=" * 70)