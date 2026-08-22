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

st.sidebar.markdown("### Navigation")
st.sidebar.markdown("Dashboard  ·  Profile  ·  Skill gaps  ·  Learning path  ·  Assistant")

available = metric_value(path, "Course Status", "Course Available")
gap_count = metric_value(path, "Course Status", "Skill Gap - No Course")
high_count = metric_value(path, "Priority", "High")
progress = available / len(path) if len(path) else 0

metric_columns = st.columns(4)
metric_columns[0].metric("Learning steps", len(path))
metric_columns[1].metric("Courses available", available)
metric_columns[2].metric("Unavailable courses", gap_count)
metric_columns[3].metric("High-priority steps", high_count)

st.progress(progress, text=f"{available} of {len(path)} steps have a course recommendation")

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
    st.dataframe(path, use_container_width=True, hide_index=True)
    st.download_button("Download learning path CSV", path.to_csv(index=False), "final_personalized_learning_path.csv", "text/csv")

with quiz_tab:
    st.subheader("Quiz and practice plan")
    quiz_columns = ["Sequence", "Stage", "Skill", "Priority", "Quiz Type", "Number of Questions", "Practice Level", "Quiz Status"]
    st.dataframe(path[quiz_columns], use_container_width=True, hide_index=True)

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
