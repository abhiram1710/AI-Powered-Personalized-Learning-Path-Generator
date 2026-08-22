"""Streamlit dashboard for the personalized learning path project."""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "Processed" / "RAG"
PROFILE_PATH = DATA_DIR / "student_profile_agent_output.csv"
SKILL_GAP_PATH = DATA_DIR / "skill_gap_agent_output.csv"
COURSES_PATH = DATA_DIR / "course_recommendation_agent_output.csv"
FINAL_PATH = DATA_DIR / "final_personalized_learning_path.csv"
INDEX_PATH = DATA_DIR / "faiss_index.index"
CHUNKS_PATH = DATA_DIR / "knowledge_chunks.pkl"

QUESTION_BANK = {
    "Python": [
        ("Which keyword defines a function in Python?", ["func", "def", "function", "lambda"], "def"),
        ("What data type stores an ordered, mutable collection?", ["Tuple", "Set", "List", "Dictionary"], "List"),
        ("Which library is commonly used for tabular data analysis?", ["Pandas", "Flask", "Pytest", "Tkinter"], "Pandas"),
    ],
    "Sql": [
        ("Which clause filters rows in a SQL query?", ["ORDER BY", "WHERE", "GROUP BY", "JOIN"], "WHERE"),
        ("Which command reads data from a table?", ["SELECT", "INSERT", "UPDATE", "DELETE"], "SELECT"),
        ("Which keyword combines rows from related tables?", ["MERGE", "JOIN", "LINK", "UNION ALL"], "JOIN"),
    ],
    "Data Preparation": [
        ("What is a common first step when preparing raw data?", ["Ignore missing values", "Inspect the data", "Deploy a model", "Delete all rows"], "Inspect the data"),
        ("Which technique handles missing numeric values?", ["Imputation", "Encryption", "Rendering", "Compilation"], "Imputation"),
        ("Why encode categorical values?", ["To make them readable", "To make them usable by models", "To remove all features", "To increase file size"], "To make them usable by models"),
    ],
    "Data Visualization": [
        ("What is a bar chart useful for comparing?", ["Categories", "Passwords", "Source code", "File permissions"], "Categories"),
        ("Which chart commonly shows a trend over time?", ["Line chart", "Pie chart", "Word cloud", "Gauge"], "Line chart"),
        ("What should a clear chart include?", ["A title and labels", "Only colors", "Hidden axes", "Unsorted data"], "A title and labels"),
    ],
    "Machine Learning": [
        ("What is supervised learning trained with?", ["Labeled data", "Only images", "No data", "Random passwords"], "Labeled data"),
        ("What does a test set evaluate?", ["Generalization", "File size", "User login", "Database speed"], "Generalization"),
        ("Which task predicts a continuous value?", ["Regression", "Classification", "Clustering", "Sorting"], "Regression"),
    ],
    "Data Mining": [
        ("What is the goal of data mining?", ["Find useful patterns", "Format documents", "Install software", "Delete records"], "Find useful patterns"),
        ("Which method groups similar records?", ["Clustering", "Encryption", "Parsing", "Indexing"], "Clustering"),
    ],
    "Big Data": [
        ("Big data is commonly described using volume, velocity, and what?", ["Variety", "Visibility", "Validity only", "Version"], "Variety"),
        ("What is horizontal scaling?", ["Adding more machines", "Buying a larger monitor", "Reducing rows", "Compressing a chart"], "Adding more machines"),
    ],
    "Hadoop": [
        ("Which Hadoop component stores distributed files?", ["HDFS", "YARN", "Hive", "Pig"], "HDFS"),
        ("What does YARN help manage?", ["Cluster resources", "Chart colors", "SQL syntax", "Passwords"], "Cluster resources"),
    ],
    "Apache Spark": [
        ("What is Apache Spark primarily used for?", ["Distributed data processing", "Graphic design", "Email hosting", "Password storage"], "Distributed data processing"),
        ("Which Spark abstraction is a distributed dataset?", ["RDD", "HTML", "CSS", "JSON schema"], "RDD"),
    ],
    "Aws": [
        ("What is AWS?", ["A cloud platform", "A database language", "A chart type", "A Python package"], "A cloud platform"),
        ("Which AWS service provides object storage?", ["S3", "EC2", "Lambda", "RDS"], "S3"),
    ],
    "Tableau": [
        ("Tableau is primarily used for what?", ["Data visualization", "Compiling Python", "Managing passwords", "Building operating systems"], "Data visualization"),
        ("What does a Tableau dashboard combine?", ["Visualizations", "Source files only", "Database servers", "Command shells"], "Visualizations"),
    ],
    "Statistical Software": [
        ("What does a mean summarize?", ["Central tendency", "File permissions", "Network speed", "Text length only"], "Central tendency"),
        ("What does a p-value help assess?", ["Evidence against a null hypothesis", "Chart dimensions", "Database size", "Missing filenames"], "Evidence against a null hypothesis"),
    ],
    "Communication": [
        ("What improves communication with stakeholders?", ["Clear, audience-aware language", "More jargon", "Hidden assumptions", "Unlabeled charts"], "Clear, audience-aware language"),
        ("What is active listening?", ["Understanding before responding", "Interrupting quickly", "Avoiding questions", "Repeating unrelated facts"], "Understanding before responding"),
    ],
    "Project Management": [
        ("What helps a project track planned work?", ["A schedule", "A password list", "A color palette", "A random sample"], "A schedule"),
        ("What should a project risk include?", ["Likelihood and impact", "Only a title", "A font choice", "A database password"], "Likelihood and impact"),
    ],
}

st.set_page_config(page_title="Personalized Learning Path", page_icon="LP", layout="wide")


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_rag_assets():
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("rb") as handle:
        chunks = pickle.load(handle)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model, np


def retrieve_knowledge(query, top_k=4):
    try:
        index, chunks, model, np = load_rag_assets()
        vector = model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(vector, top_k)
        return [
            {"score": float(distance), "content": chunks[index]["content"], "source": chunks[index].get("source", "knowledge base")}
            for distance, index in zip(distances[0], indices[0])
            if index >= 0 and index < len(chunks)
        ]
    except Exception as error:
        return [{"error": str(error)}]


def metric_value(dataframe, column, value):
    return int((dataframe[column] == value).sum())


def quiz_questions(skill):
    return QUESTION_BANK.get(skill, [
        (f"Which action best supports progress in {skill}?", ["Practice the skill", "Skip all practice", "Hide the result", "Delete the notes"], "Practice the skill"),
        (f"What is a useful way to learn {skill}?", ["Apply it to a small problem", "Avoid examples", "Ignore feedback", "Use no resources"], "Apply it to a small problem"),
    ])


def progress_key(student_id, sequence, skill):
    return f"{student_id}:{sequence}:{skill}"


st.markdown("# Personalised Learning Path")
st.caption("A data-driven study dashboard for the next step in your career journey.")

try:
    profile = load_csv(PROFILE_PATH)
    skill_gaps = load_csv(SKILL_GAP_PATH)
    courses = load_csv(COURSES_PATH)
    final_path = load_csv(FINAL_PATH)
except (FileNotFoundError, pd.errors.ParserError) as error:
    st.error(f"Project data could not be loaded: {error}")
    st.stop()

student_ids = profile["Student ID"].dropna().astype(str).tolist()
selected_id = st.sidebar.selectbox("Student ID", student_ids)
student_profile = profile[profile["Student ID"].astype(str) == selected_id].iloc[0]
path = final_path[final_path["Student ID"].astype(str) == selected_id].copy()
gaps = skill_gaps[skill_gaps["Student ID"].astype(str) == selected_id]
student_courses = courses[courses["Student ID"].astype(str) == selected_id]

if "learning_progress" not in st.session_state:
    st.session_state.learning_progress = {}
progress_state = st.session_state.learning_progress.setdefault(selected_id, {})
for _, step in path.iterrows():
    key = progress_key(selected_id, step["Sequence"], step["Skill"])
    progress_state.setdefault(key, "Not Started")

st.sidebar.markdown("### Navigation")
st.sidebar.markdown("Dashboard  ·  Profile  ·  Skill gaps  ·  Learning path  ·  Assistant")

available = metric_value(path, "Course Status", "Course Available")
gap_count = metric_value(path, "Course Status", "Skill Gap - No Course")
high_count = metric_value(path, "Priority", "High")
completed_count = sum(value == "Completed" for value in progress_state.values())
progress = completed_count / len(path) if len(path) else 0

metric_columns = st.columns(4)
metric_columns[0].metric("Learning steps", len(path))
metric_columns[1].metric("Courses available", available)
metric_columns[2].metric("Unavailable courses", gap_count)
metric_columns[3].metric("Completed steps", f"{completed_count}/{len(path)}")

st.progress(progress, text=f"{completed_count} of {len(path)} learning steps completed")

profile_tab, gaps_tab, courses_tab, path_tab, quiz_tab, assistant_tab = st.tabs(
    ["Student Profile", "Skill Gap Analysis", "Recommended Courses", "Learning Path", "Quiz & Practice", "RAG Assistant"]
)

with profile_tab:
    st.subheader("Student profile")
    left, right = st.columns([1, 2])
    with left:
        st.write(f"**Student ID:** {selected_id}")
        st.write(f"**Profession:** {student_profile.get('Job Profession', 'Not available')}")
        st.write(f"**Dominant intelligence:** {student_profile.get('Dominant Intelligence', 'Not available')}")
        st.write(f"**Intelligence score:** {student_profile.get('Intelligence Score', 'Not available')}")
    with right:
        intelligence_columns = ["Linguistic", "Musical", "Bodily", "Logical Mathematical", "Spatial Visualization", "Interpersonal", "Intrapersonal", "Naturalist"]
        score_data = pd.DataFrame({"Intelligence": intelligence_columns, "Score": [student_profile.get(column, 0) for column in intelligence_columns]})
        st.bar_chart(score_data.set_index("Intelligence"), horizontal=True)

with gaps_tab:
    st.subheader("Skill gap analysis")
    gap_metrics = st.columns(5)
    gap_metrics[0].metric("Total gaps", len(gaps))
    gap_metrics[1].metric("High priority", metric_value(gaps, "Priority", "High"))
    gap_metrics[2].metric("Medium priority", metric_value(gaps, "Priority", "Medium"))
    gap_metrics[3].metric("Course available", available)
    gap_metrics[4].metric("Course unavailable", gap_count)
    st.dataframe(gaps, use_container_width=True, hide_index=True)

with courses_tab:
    st.subheader("Recommended courses")
    recommended = student_courses[student_courses["Course Status"] == "Course Recommended"]
    if recommended.empty:
        st.info("No course recommendations are available for this student.")
    else:
        st.dataframe(recommended, use_container_width=True, hide_index=True)
    unavailable_skills = student_courses[student_courses["Course Status"] == "No Course Available"]["Skill"].tolist()
    if unavailable_skills:
        st.warning("Courses unavailable for: " + ", ".join(unavailable_skills))

with path_tab:
    st.subheader("Ordered personalized learning path")
    st.caption("Progress is stored for this browser session only.")
    progress_table = path[["Sequence", "Stage", "Skill", "Priority", "Course Name", "Course Status", "Quiz Status"]].copy()
    progress_table["Progress"] = [
        progress_state[progress_key(selected_id, row["Sequence"], row["Skill"])]
        for _, row in progress_table.iterrows()
    ]
    edited_progress = st.data_editor(
        progress_table,
        use_container_width=True,
        hide_index=True,
        disabled=["Sequence", "Stage", "Skill", "Priority", "Course Name", "Course Status", "Quiz Status"],
        column_config={
            "Progress": st.column_config.SelectboxColumn(
                "Progress", options=["Not Started", "In Progress", "Completed"], required=True
            )
        },
        key=f"progress_editor_{selected_id}",
    )
    for _, row in edited_progress.iterrows():
        progress_state[progress_key(selected_id, row["Sequence"], row["Skill"])] = row["Progress"]
    updated_completed = sum(value == "Completed" for value in progress_state.values())
    st.write(f"**{updated_completed} / {len(path)} steps completed ({updated_completed / len(path):.0%})**")
    st.download_button("Download learning path CSV", path.to_csv(index=False), "final_personalized_learning_path.csv", "text/csv")

with quiz_tab:
    st.subheader("Quiz and practice plan")
    quiz_columns = ["Sequence", "Stage", "Skill", "Priority", "Quiz Type", "Number of Questions", "Practice Level", "Quiz Status"]
    st.dataframe(path[quiz_columns], use_container_width=True, hide_index=True)
    if not path.empty:
        selected_sequence = st.selectbox(
            "Select a learning step", path["Sequence"].tolist(),
            format_func=lambda sequence: f"Step {sequence}: {path.loc[path['Sequence'] == sequence, 'Skill'].iloc[0]}"
        )
        selected_step = path[path["Sequence"] == selected_sequence].iloc[0]
        st.markdown(
            f"**{selected_step['Skill']}** · {selected_step['Stage']} · "
            f"{selected_step['Priority']} priority"
        )
        st.write(
            f"Quiz type: {selected_step['Quiz Type']}  |  "
            f"Questions: {selected_step['Number of Questions']}  |  "
            f"Practice: {selected_step['Practice Level']}"
        )
        if selected_step["Quiz Status"] == "Practice Required - No Course":
            st.warning("No course is currently available for this skill. Practice is still recommended.")
        questions = quiz_questions(selected_step["Skill"])
        with st.form(f"quiz_form_{selected_id}_{selected_sequence}"):
            answers = [
                st.radio(question, options, key=f"answer_{selected_id}_{selected_sequence}_{index}")
                for index, (question, options, _) in enumerate(questions)
            ]
            submitted = st.form_submit_button("Submit quiz")
        if submitted:
            score = sum(answer == correct for answer, (_, _, correct) in zip(answers, questions))
            percentage = score / len(questions) * 100
            feedback = "Excellent" if percentage >= 80 else "Good" if percentage >= 50 else "Needs Practice"
            st.success(f"Score: {score}/{len(questions)} ({percentage:.0f}%) · {feedback}")

with assistant_tab:
    st.subheader("Ask about my learning path")
    question = st.text_input("Question", placeholder="What should I learn first?")
    if question:
        with st.spinner("Searching the learning knowledge base..."):
            results = retrieve_knowledge(question)
        if results and "error" not in results[0]:
            st.markdown("#### Knowledge-base context")
            for result in results:
                with st.expander(f"{result['source']} · score {result['score']:.3f}"):
                    st.write(result["content"])
            first_skill = path.sort_values("Sequence")["Skill"].iloc[0] if not path.empty else "your first listed skill"
            st.info(f"Your ordered path starts with **{first_skill}**. Use the retrieved context above to guide your next study session.")
        else:
            message = results[0].get("error", "No relevant knowledge was found.") if results else "No relevant knowledge was found."
            st.warning(f"The knowledge assistant is unavailable right now: {message}")

st.caption("Generated from the completed Agent 1-6 pipeline and project knowledge base.")
