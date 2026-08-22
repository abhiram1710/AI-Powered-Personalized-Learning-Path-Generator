"""Authenticated Streamlit dashboard for the personalized learning platform."""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

import auth_db


ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / "Dataset" / "Processed" / "RAG"
PROFILE_PATH = RAG_DIR / "student_profile_agent_output.csv"
FINAL_PATH = RAG_DIR / "final_personalized_learning_path.csv"
GAPS_PATH = RAG_DIR / "skill_gap_agent_output.csv"
COURSES_PATH = RAG_DIR / "course_recommendation_agent_output.csv"
RANKED_PATH = RAG_DIR / "ranked_courses.csv"
INDEX_PATH = RAG_DIR / "faiss_index.index"
CHUNKS_PATH = RAG_DIR / "knowledge_chunks.pkl"

QUESTIONS = {
    "Python": [("Which keyword defines a function?", ["func", "def", "lambda", "method"], "def")],
    "Sql": [("Which clause filters rows?", ["WHERE", "ORDER BY", "JOIN", "GROUP BY"], "WHERE")],
    "JavaScript": [("Which keyword declares a block-scoped variable?", ["let", "var", "define", "dim"], "let")],
    "Machine Learning": [("Which task predicts a continuous value?", ["Regression", "Classification", "Clustering", "Sorting"], "Regression")],
    "Aws": [("Which AWS service provides object storage?", ["S3", "EC2", "IAM", "RDS"], "S3")],
}

st.set_page_config(page_title="Personalised Learning Platform", page_icon="LP", layout="wide")
auth_db.init_db()


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def rag_assets():
    import faiss
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("rb") as handle:
        chunks = pickle.load(handle)
    return index, chunks, SentenceTransformer("all-MiniLM-L6-v2")


def rag_search(query, user):
    try:
        index, chunks, model = rag_assets()
        context = f"Career goal: {user['career_goal']}. Current skills: {user['current_skills']}. Interests: {user['interests']}. {query}"
        vector = model.encode([context], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(vector, 4)
        return [{"score": float(score), "content": chunks[index]["content"], "source": chunks[index].get("source", "knowledge base")} for score, index in zip(distances[0], indices[0]) if 0 <= index < len(chunks)]
    except Exception as error:
        return [{"error": str(error)}]


def auth_page():
    st.title("Personalised Learning Platform")
    login_tab, register_tab = st.tabs(["Login", "Create account"])
    with login_tab:
        identifier = st.text_input("Username or email", key="login_identifier")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            user = auth_db.authenticate(identifier, password)
            if user:
                st.session_state.user_id = user["id"]
                st.rerun()
            st.error("Invalid username/email or password.")
    with register_tab:
        with st.form("register_form"):
            full_name = st.text_input("Full name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            career = st.text_input("Career goal", placeholder="Data Scientist")
            interests = st.text_input("Areas of interest", placeholder="Python, SQL, Machine Learning")
            skills = st.text_input("Current skills", placeholder="Python")
            level = st.selectbox("Preferred learning level", ["Beginner", "Intermediate", "Advanced"])
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    user_id = auth_db.register_user(full_name, username, email, password, career, interests, skills, level)
                    new_user = auth_db.get_user(user_id)
                    ranked = read_csv(RANKED_PATH)
                    auth_db.save_personalization(user_id, new_user, ranked, auth_db.build_user_path(new_user, ranked))
                    st.success("Account created. You can now log in.")
                except ValueError as error:
                    st.error(str(error))


def sample_dashboard():
    profile = read_csv(PROFILE_PATH)
    path = read_csv(FINAL_PATH)
    gaps = read_csv(GAPS_PATH)
    courses = read_csv(COURSES_PATH)
    return profile.iloc[0].to_dict(), path, gaps, courses


def user_dashboard(user):
    ranked = read_csv(RANKED_PATH)
    data = auth_db.user_data(user["id"])
    path = pd.DataFrame(data["learning_path"])
    gaps = pd.DataFrame(data["skill_gaps"])
    courses = pd.DataFrame(data["course_recommendations"])
    return {"Student ID": user["username"], "Student": user["full_name"], "Job Profession": user["career_goal"], "Dominant Intelligence": "Personalized", "Career Goal": user["career_goal"], "Interests": user["interests"]}, path, gaps, courses


def dashboard(user):
    if user.get("id") == 0:
        profile, path, gaps, courses = sample_dashboard()
        dynamic = False
    else:
        profile, path, gaps, courses = user_dashboard(user)
        dynamic = True
    st.title(f"Welcome, {profile.get('Student', user['full_name'])}")
    st.caption(f"Career goal: {profile.get('Career Goal', profile.get('Job Profession', 'Not specified'))}  ·  Interests: {profile.get('Interests', 'Project reference profile')}")
    if dynamic:
        st.info("Your registered profile is isolated to your account. Progress and quiz results are stored in SQLite.")
    available = int((path["course_status"] == "Course Available").sum()) if dynamic and not path.empty else int((path["Course Status"] == "Course Available").sum())
    total = len(path)
    completed = int((path["progress_status"] == "Completed").sum()) if dynamic and not path.empty else 0
    metrics = st.columns(4)
    metrics[0].metric("Learning steps", total)
    metrics[1].metric("Courses", available)
    metrics[2].metric("Skill gaps", len(gaps))
    metrics[3].metric("Progress", f"{completed}/{total}")
    st.progress(completed / total if total else 0, text=f"{completed} of {total} steps completed")

    profile_tab, gap_tab, course_tab, path_tab, quiz_tab, rag_tab = st.tabs(["My Profile", "Skill Analysis", "Courses", "Learning Path", "Quizzes", "RAG Assistant"])
    with profile_tab:
        left, right = st.columns([1, 2])
        with left:
            for key in ("Student", "Student ID", "Job Profession", "Career Goal", "Dominant Intelligence", "Intelligence Score"):
                if key in profile: st.write(f"**{key}:** {profile[key]}")
        with right:
            intelligence = [column for column in ("Linguistic", "Musical", "Bodily", "Logical Mathematical", "Spatial Visualization", "Interpersonal", "Intrapersonal", "Naturalist") if column in profile]
            if intelligence: st.bar_chart(pd.DataFrame({"Intelligence": intelligence, "Score": [profile[column] for column in intelligence]}).set_index("Intelligence"), horizontal=True)
    with gap_tab:
        st.metric("Total skill gaps", len(gaps))
        st.dataframe(gaps, use_container_width=True, hide_index=True)
    with course_tab:
        st.dataframe(courses, use_container_width=True, hide_index=True)
        if dynamic and courses.empty: st.info("No course recommendations are available yet.")
    with path_tab:
        if dynamic and not path.empty:
            for _, step in path.iterrows():
                status = st.selectbox(f"Step {step['sequence']}: {step['skill']}", ["Not Started", "In Progress", "Completed"], index=["Not Started", "In Progress", "Completed"].index(step["progress_status"]), key=f"status_{step['id']}")
                if status != step["progress_status"]: auth_db.update_progress(user["id"], step["id"], status); st.rerun()
            st.dataframe(path, use_container_width=True, hide_index=True)
        else: st.dataframe(path, use_container_width=True, hide_index=True)
    with quiz_tab:
        if path.empty: st.info("No learning steps available.")
        else:
            skill_column = "skill" if dynamic else "Skill"
            selected_skill = st.selectbox("Select learning skill", path[skill_column].tolist())
            step = path[path[skill_column] == selected_skill].iloc[0]
            stage = step.get("stage", step.get("Stage", ""))
            priority = step.get("priority", step.get("Priority", ""))
            st.write(f"**{selected_skill}** · {stage} · {priority} priority")
            questions = QUESTIONS.get(selected_skill, [(f"What is a useful first step for {selected_skill}?", ["Practice the skill", "Skip practice", "Ignore feedback", "Use no resources"], "Practice the skill")])
            with st.form(f"quiz_{user.get('id', 0)}_{selected_skill}"):
                answers = [st.radio(question, options) for question, options, _ in questions]
                submit = st.form_submit_button("Submit quiz")
            if submit:
                score = sum(answer == correct for answer, (_, _, correct) in zip(answers, questions)); percentage = score / len(questions) * 100
                if dynamic:
                    with auth_db.connect() as db:
                        quiz = db.execute("SELECT id FROM quizzes WHERE user_id=? AND skill=? ORDER BY id LIMIT 1", (user["id"], selected_skill)).fetchone()
                    if quiz is not None:
                        auth_db.save_quiz_result(user["id"], quiz["id"], score, len(questions))
                    st.write(f"Quiz result saved: {score}/{len(questions)} ({percentage:.0f}%)")
                else: st.success(f"Score: {score}/{len(questions)} ({percentage:.0f}%)")
    with rag_tab:
        question = st.text_input("Ask about your learning path", placeholder="What should I learn next?")
        if question:
            results = rag_search(question, user)
            if results and "error" not in results[0]:
                for result in results: st.expander(f"{result['source']} · {result['score']:.3f}").write(result["content"])
                next_skill = path.iloc[0].get("skill", path.iloc[0].get("Skill", "your next skill")) if not path.empty else "your next skill"
                st.info(f"Based on your profile and path, focus next on **{next_skill}**.")
            else: st.warning(f"RAG assistant unavailable: {results[0].get('error', 'No results') if results else 'No results'}")


if "user_id" not in st.session_state:
    auth_page()
else:
    with auth_db.connect() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (st.session_state.user_id,)).fetchone()
    if row is None:
        st.session_state.pop("user_id")
        st.rerun()
    user = dict(row)
    with st.sidebar:
        st.write(f"Signed in as **{user['full_name']}**")
        if st.button("Logout"): st.session_state.clear(); st.rerun()
        st.caption("Progress is stored per account in SQLite.")
    dashboard(user)
