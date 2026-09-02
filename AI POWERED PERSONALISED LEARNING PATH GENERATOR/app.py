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

CONCEPT_MATERIALS = {
    "Python Variables and data types": ("Python data structures", "https://docs.python.org/3/tutorial/introduction.html"),
    "Python Collections": ("Python data structures", "https://docs.python.org/3/tutorial/datastructures.html"),
    "Python Dictionaries": ("Python dictionaries", "https://docs.python.org/3/tutorial/datastructures.html#dictionaries"),
    "Python Functions": ("Python functions", "https://docs.python.org/3/tutorial/controlflow.html#defining-functions"),
    "Python Exception handling": ("Python exceptions", "https://docs.python.org/3/tutorial/errors.html"),
    "Python Iteration": ("Python control flow", "https://docs.python.org/3/tutorial/controlflow.html"),
    "Python Object-oriented programming": ("Python classes", "https://docs.python.org/3/tutorial/classes.html"),
    "Python Modules": ("Python modules", "https://docs.python.org/3/tutorial/modules.html"),
    "Python Built-in functions": ("Python built-in functions", "https://docs.python.org/3/library/functions.html"),
    "Python Special values": ("Python built-in constants", "https://docs.python.org/3/library/constants.html"),
    "Python Comments": ("Python introduction", "https://docs.python.org/3/tutorial/introduction.html"),
    "Pandas Series and DataFrame": ("Pandas intro to data structures", "https://pandas.pydata.org/docs/user_guide/dsintro.html"),
    "Pandas Indexing": ("Pandas indexing", "https://pandas.pydata.org/docs/user_guide/indexing.html"),
    "Pandas Filtering": ("Pandas indexing and selecting data", "https://pandas.pydata.org/docs/user_guide/indexing.html"),
    "Pandas GroupBy": ("Pandas GroupBy", "https://pandas.pydata.org/docs/user_guide/groupby.html"),
    "Pandas Merge and join": ("Pandas merge", "https://pandas.pydata.org/docs/user_guide/merging.html"),
    "Pandas Missing values": ("Pandas missing data", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),
    "Pandas Sorting": ("Pandas sorting", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html"),
    "Pandas Pivot tables": ("Pandas pivot tables", "https://pandas.pydata.org/docs/user_guide/reshaping.html"),
    "Pandas Apply and map": ("Pandas function application", "https://pandas.pydata.org/docs/user_guide/basics.html#function-application"),
    "Pandas Reading CSV": ("Pandas read_csv", "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html"),
    "Sql WHERE filtering": ("SQL WHERE", "https://www.postgresql.org/docs/current/tutorial-select.html"),
    "Sql SELECT": ("SQL SELECT", "https://www.postgresql.org/docs/current/tutorial-select.html"),
    "Sql JOINs": ("SQL joins", "https://www.postgresql.org/docs/current/tutorial-join.html"),
    "Sql Aggregate functions": ("SQL aggregate functions", "https://www.postgresql.org/docs/current/functions-aggregate.html"),
    "Sql AVG function": ("PostgreSQL aggregate functions", "https://www.postgresql.org/docs/current/functions-aggregate.html"),
    "Sql GROUP BY": ("SQL GROUP BY", "https://www.postgresql.org/docs/current/tutorial-agg.html"),
    "Sql HAVING": ("SQL HAVING", "https://www.postgresql.org/docs/current/tutorial-agg.html"),
    "Sql Primary keys": ("PostgreSQL constraints", "https://www.postgresql.org/docs/current/ddl-constraints.html"),
    "Sql INSERT": ("SQL INSERT", "https://www.postgresql.org/docs/current/dml-insert.html"),
    "Sql Subqueries": ("SQL subqueries", "https://www.postgresql.org/docs/current/functions-subquery.html"),
    "Machine Learning Supervised learning": ("Scikit-learn supervised learning", "https://scikit-learn.org/stable/supervised_learning.html"),
    "Machine Learning Regression": ("Scikit-learn linear models", "https://scikit-learn.org/stable/modules/linear_model.html"),
    "Machine Learning Classification": ("Scikit-learn classification", "https://scikit-learn.org/stable/supervised_learning.html"),
    "Machine Learning Train/test split": ("Scikit-learn model selection", "https://scikit-learn.org/stable/model_selection.html"),
    "Machine Learning Overfitting": ("Scikit-learn learning curves", "https://scikit-learn.org/stable/modules/learning_curve.html"),
    "Machine Learning Evaluation metrics": ("Scikit-learn metrics", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
    "Machine Learning Features": ("Scikit-learn preprocessing", "https://scikit-learn.org/stable/modules/preprocessing.html"),
    "Machine Learning Clustering": ("Scikit-learn clustering", "https://scikit-learn.org/stable/modules/clustering.html"),
    "Machine Learning Feature scaling": ("Scikit-learn scaling", "https://scikit-learn.org/stable/modules/preprocessing.html"),
    "Machine Learning Predictions": ("Scikit-learn model evaluation", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
}

CONCEPT_EXPLANATIONS = {
    "Variables and data types": "Variables refer to values, and data types describe what operations are valid for those values.",
    "Collections": "Lists are ordered and mutable, while dictionaries map keys to values; choosing the right collection affects access and updates.",
    "Dictionaries": "A dictionary stores key-value pairs and is useful when values should be retrieved by a meaningful key.",
    "Functions": "Functions package reusable behavior behind a name and can receive arguments and return values.",
    "Exception handling": "try/except catches expected runtime errors so a program can respond instead of stopping unexpectedly.",
    "Iteration": "A for loop visits items in an iterable one at a time, making repeated work explicit and readable.",
    "Object-oriented programming": "A class is a blueprint for objects that combines related data and behavior.",
    "Modules": "import loads code from a module so related functionality can be reused without copying it.",
    "Built-in functions": "Built-in functions such as len() are available without importing a module and provide common operations.",
    "Special values": "None represents the absence of a value and is distinct from numeric zero or an empty collection.",
    "Comments": "The # marker begins a Python comment, which documents code and is ignored by the interpreter.",
    "Series and DataFrame": "A Series is one-dimensional; a DataFrame is a two-dimensional labeled table made from aligned Series.",
    "Indexing": "loc selects by labels, while iloc selects by integer positions.",
    "Filtering": "Boolean indexing keeps only rows whose condition evaluates to True.",
    "GroupBy": "groupby divides rows into groups and lets you apply aggregations such as sum, mean, or count.",
    "Merge and join": "merge combines DataFrames using matching keys; it is different from grouping rows for aggregation.",
    "Missing values": "fillna replaces missing values using a chosen value or strategy, while dropna removes affected rows or columns.",
    "Sorting": "sort_values orders rows by one or more columns without changing the meaning of the values.",
    "Pivot tables": "pivot_table summarizes data across row and column dimensions using an aggregation function.",
    "Apply and map": "map applies a function element-wise to a Series, while apply supports broader row, column, or element operations.",
    "Reading CSV": "read_csv parses a CSV file into a DataFrame so its columns can be inspected and transformed.",
    "WHERE filtering": "WHERE filters individual rows before grouping or aggregation.",
    "SELECT": "SELECT specifies the columns and expressions returned from a query.",
    "JOINs": "A JOIN combines rows from related tables using a matching relationship or key.",
    "Aggregate functions": "COUNT, SUM, and AVG summarize multiple rows into a calculated result.",
    "AVG function": "AVG calculates the arithmetic mean of non-null input values in a SQL query.",
    "GROUP BY": "GROUP BY forms groups so aggregate functions can calculate one result per group.",
    "HAVING": "HAVING filters grouped results after aggregation, unlike WHERE which filters rows first.",
    "Primary keys": "A primary key uniquely identifies each row and prevents duplicate identity values.",
    "INSERT": "INSERT adds new rows to a table and supplies values for its columns.",
    "Subqueries": "A subquery is a query nested inside another query and can provide values or rows for the outer query.",
    "Supervised learning": "Supervised learning learns a mapping from labeled examples to predict labels or numeric targets.",
    "Regression": "Regression predicts continuous numeric values, such as a price or temperature.",
    "Classification": "Classification predicts a discrete class, such as spam or not spam.",
    "Train/test split": "A train/test split evaluates a model on held-out examples rather than only on data it saw during training.",
    "Overfitting": "Overfitting happens when a model memorizes training detail and performs poorly on unseen data.",
    "Evaluation metrics": "Metrics quantify model performance; the right metric depends on the task and its costs.",
    "Features": "Features are input variables used by a model to produce a prediction.",
    "Clustering": "Clustering groups similar unlabeled observations without requiring predefined target labels.",
    "Feature scaling": "Scaling puts numeric features on comparable ranges so magnitude does not dominate some algorithms.",
    "Predictions": "A prediction is the model output produced from input features after learning from training data.",
}


def review_details(skill, concept, question, user_answer, correct_answer):
    material_key = f"{skill} {concept}"
    material = next(
        (value for key, value in CONCEPT_MATERIALS.items()
         if "".join(character.lower() for character in key if character.isalnum())
         == "".join(character.lower() for character in material_key if character.isalnum())),
        None,
    )
    explanation = CONCEPT_EXPLANATIONS.get(concept)
    if material is None:
        material = (f"{skill} official learning material", {
            "Python": "https://docs.python.org/3/tutorial/",
            "Pandas": "https://pandas.pydata.org/docs/user_guide/",
            "Sql": "https://www.postgresql.org/docs/current/tutorial-sql.html",
            "Machine Learning": "https://developers.google.com/machine-learning/crash-course",
        }.get(skill, "https://www.khanacademy.org/"))
    if not explanation:
        explanation = f"Review the core {skill} concept and apply it in a small practical exercise."
    material_title, material_url = material
    return explanation, material_title, material_url


def enrich_quiz_rows(skill, quiz_rows):
    """Attach and validate review metadata before a quiz can be displayed."""
    enriched = []
    legacy_concept = " ".join(("Core", "concept"))
    for row in quiz_rows:
        concept = str(row.get("concept", "")).strip()
        if not concept or concept.casefold() in {legacy_concept.casefold(), "unassigned"}:
            raise ValueError(f"Quiz data error: missing explicit concept for {skill}.")
        explanation, material_title, material_url = review_details(
            skill, concept, row["question"], "", row["correct_answer"]
        )
        if not explanation or not material_url:
            raise ValueError(f"Quiz data error: incomplete metadata for {skill} / {concept}.")
        enriched_row = dict(row)
        enriched_row.update({
            "explanation": explanation,
            "material_title": material_title,
            "material_url": material_url,
        })
        enriched.append(enriched_row)
    return enriched

st.set_page_config(page_title="Personalised Learning Platform", page_icon="LP", layout="wide")
auth_db.init_db()


@st.cache_data
def read_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def rag_index_and_chunks():
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    with CHUNKS_PATH.open("rb") as handle:
        chunks = pickle.load(handle)
    return index, chunks


@st.cache_resource
def rag_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def rag_search(query, user):
    try:
        index, chunks = rag_index_and_chunks()
        model = rag_embedding_model()
        context = f"Career goal: {user['career_goal']}. Current skills: {user['current_skills']}. Interests: {user['interests']}. Question: {query}"
        vector = model.encode([context], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(vector, 4)
        results = [{"score": float(score), "content": chunks[index]["content"], "source": chunks[index].get("source", "knowledge base")} for score, index in zip(distances[0], indices[0]) if 0 <= index < len(chunks) and str(chunks[index].get("content", "")).strip()]
        return results
    except Exception as error:
        return [{"error": f"RAG assets are unavailable: {error}"}]


def grounded_answer(question, results, path, user):
    """Generate only from retrieved context and the current user's path."""
    if not results:
        return "I couldn't find relevant information in the knowledge base for this question."
    if "error" in results[0]:
        return results[0]["error"]
    remaining = path[path["progress_status"] != "Completed"] if not path.empty and "progress_status" in path else path
    path_context = remaining[[column for column in ("sequence", "skill", "skill_category", "course_name", "course_status", "progress_status") if column in remaining]].to_dict("records") if not remaining.empty else []
    retrieved_context = "\n\n".join(result["content"] for result in results)
    prompt = f"""User Question:\n{question}\n\nRetrieved Context:\n{retrieved_context}\n\nUser Learning Path:\n{path_context}\n\nInstructions:\n- Answer using the retrieved context and the user's learning path.\n- Do not invent information.\n- Keep the answer relevant and concise.\n- If the retrieved context does not contain the answer, say that the information was not found.\n- For learning-order questions, use the first incomplete path step and its prerequisite order, not unrelated skills."""
    try:
        secret_config = st.secrets.get("rag", {})
        api_key = secret_config.get("api_key", "")
        model_name = secret_config.get("model", "gpt-4o-mini")
    except Exception:
        api_key, model_name = "", "gpt-4o-mini"
    if api_key:
        try:
            from openai import OpenAI
            response = OpenAI(api_key=api_key, timeout=15.0, max_retries=0).chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            return response.choices[0].message.content.strip()
        except Exception as error:
            return f"The grounded language model could not answer this question: {error}"
    is_learning_path_question = any(keyword in question.casefold() for keyword in ("learn", "learning", "skill", "course", "path", "next", "start", "career"))
    if path_context and is_learning_path_question:
        next_step = path_context[0]
        return f"Your next incomplete learning step is **{next_step['skill']}**. The knowledge base context supports this path position, but no language-model API key is configured for a synthesized explanation."
    return "I couldn't find relevant information in the knowledge base for this question."

def auth_page():
    st.title("Personalised Learning Platform")

    # Get reset token from email link
    url_reset_token = st.query_params.get("reset_token", "")

    # ---------------------------------------------------------
    # PASSWORD RESET PAGE
    # ---------------------------------------------------------
    if url_reset_token:
        st.subheader("Reset your password")
        st.info("Create a new password for your account.")

        new_password = st.text_input(
            "New password",
            type="password",
            key="reset_password"
        )

        confirm_reset = st.text_input(
            "Confirm new password",
            type="password",
            key="reset_password_confirm"
        )

        if st.button(
            "Reset password",
            type="primary",
            key="reset_password_button"
        ):
            if not new_password:
                st.error("Please enter a new password.")

            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")

            elif new_password != confirm_reset:
                st.error("Passwords do not match.")

            else:
                try:
                    auth_db.reset_password(
                        url_reset_token,
                        new_password
                    )

                    st.query_params.clear()

                    st.success(
                        "Password reset successfully. You can now log in."
                    )

                except ValueError:
                    st.error(
                        "The reset link is invalid or has expired."
                    )

        return

    # ---------------------------------------------------------
    # LOGIN / CREATE ACCOUNT
    # ---------------------------------------------------------
    login_tab, register_tab = st.tabs(
        ["Login", "Create account"]
    )

    # =========================================================
    # LOGIN TAB
    # =========================================================
    with login_tab:

        identifier = st.text_input(
            "Username or email",
            key="login_identifier"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", type="primary", key="login_button"):

            user = auth_db.authenticate(
                identifier,
                password
            )

            if user:
                st.session_state.user_id = user["id"]
                st.rerun()

            else:
                st.error(
                    "Invalid username/email or password."
                )

        # -----------------------------------------------------
        # FORGOT PASSWORD
        # -----------------------------------------------------
        with st.expander("Forgot Password?"):

            reset_identifier = st.text_input(
                "Username or email",
                key="reset_identifier"
            )

            if st.button(
                "Send reset email",
                key="generate_reset"
            ):

                try:
                    auth_db.request_password_reset(
                        reset_identifier
                    )

                    st.success(
                        "If the account exists, password reset "
                        "instructions have been sent to the "
                        "registered email address."
                    )

                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    # =========================================================
    # CREATE ACCOUNT TAB
    # =========================================================
    with register_tab:

        with st.form("register_form"):

            full_name = st.text_input(
                "Full name"
            )

            username = st.text_input(
                "Username"
            )

            email = st.text_input(
                "Email"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            confirm = st.text_input(
                "Confirm password",
                type="password"
            )

            career = st.text_input(
                "Career goal",
                placeholder="Data Scientist"
            )

            interests = st.text_input(
                "Areas of interest",
                placeholder="Python, SQL, Machine Learning"
            )

            skills = st.text_input(
                "Current skills",
                placeholder="Python"
            )

            level = st.selectbox(
                "Preferred learning level",
                [
                    "Beginner",
                    "Intermediate",
                    "Advanced"
                ]
            )

            submitted = st.form_submit_button(
                "Create account",
                type="primary"
            )

        if submitted:

            if not full_name:
                st.error("Please enter your full name.")

            elif not username:
                st.error("Please enter a username.")

            elif not email:
                st.error("Please enter your email.")

            elif not password:
                st.error("Please enter a password.")

            elif len(password) < 8:
                st.error(
                    "Password must be at least 8 characters."
                )

            elif password != confirm:
                st.error(
                    "Passwords do not match."
                )

            else:

                try:

                    user_id = auth_db.register_user(
                        full_name,
                        username,
                        email,
                        password,
                        career,
                        interests,
                        skills,
                        level
                    )

                    new_user = auth_db.get_user(user_id)

                    ranked = read_csv(RANKED_PATH)

                    auth_db.save_personalization(
                        user_id,
                        new_user,
                        ranked,
                        auth_db.build_user_path(
                            new_user,
                            ranked
                        )
                    )

                    st.success(
                        "Account created successfully. "
                        "You can now log in."
                    )

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
    completed_skills = path.loc[path.get("progress_status", pd.Series(dtype=str)) == "Completed", "skill"].tolist() if not path.empty else []
    profile = {"Student ID": user["username"], "Student": user["full_name"], "Career Goal": user["career_goal"], "Interests": user["interests"], "Current Skills": user["current_skills"], "Completed Skills": ", ".join(completed_skills) or "None yet", "Learning Progress": f"{sum(path['progress_status'] == 'Completed')}/{len(path)}"}
    return profile, path, gaps, courses, data


def dashboard(user):
    if user.get("id") == 0:
        profile, path, gaps, courses = sample_dashboard()
        data = {}
        dynamic = False
    else:
        profile, path, gaps, courses, data = user_dashboard(user)
        dynamic = True
    st.title(f"Welcome, {profile.get('Student', user['full_name'])}")
    st.caption(f"Career goal: {profile.get('Career Goal', profile.get('Job Profession', 'Not specified'))}  ·  Interests: {profile.get('Interests', 'Project reference profile')}")
    if dynamic:
        st.info("Your registered profile is isolated to your account.")
    available = int((courses["course_status"].isin(["Course Available", "Recommended"])).sum()) if dynamic and not courses.empty else int((path["Course Status"] == "Course Available").sum())
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
            for key in ("Student", "Student ID", "Career Goal", "Interests", "Current Skills", "Completed Skills", "Learning Progress", "Job Profession", "Dominant Intelligence", "Intelligence Score"):
                if key in profile: st.write(f"**{key}:** {profile[key]}")
        with right:
            intelligence = [column for column in ("Linguistic", "Musical", "Bodily", "Logical Mathematical", "Spatial Visualization", "Interpersonal", "Intrapersonal", "Naturalist") if column in profile]
            if intelligence: st.bar_chart(pd.DataFrame({"Intelligence": intelligence, "Score": [profile[column] for column in intelligence]}).set_index("Intelligence"), horizontal=True)
    with gap_tab:
        if dynamic and not gaps.empty:
            gaps = gaps.assign(status=gaps["skill"].map(dict(zip(path["skill"], path["progress_status"]))))
            gaps["gap"] = gaps["status"].map(lambda status: "No Gap" if status == "Completed" else "Gap")
        st.metric("Total skill gaps", int((gaps.get("gap", pd.Series(dtype=str)) == "Gap").sum()))
        st.dataframe(gaps, use_container_width=True, hide_index=True)
    with course_tab:
        if dynamic and not path.empty:
            available_courses = courses[courses["course_status"] != "Completed"].copy() if not courses.empty else pd.DataFrame()
            available_keys = set(available_courses.get("skill_key", pd.Series(dtype=str)))
            unavailable = path[~path["skill_key"].isin(available_keys)].copy()
            if not unavailable.empty:
                unavailable["course_name"] = "Course Not Available"
                unavailable["course_status"] = "Course Not Available"
                unavailable["provider"] = ""
                unavailable["level"] = ""
                unavailable["rating"] = None
                unavailable["duration"] = ""
                unavailable["course_url"] = ""
                available_courses = pd.concat([available_courses, unavailable[["skill", "skill_key", "course_name", "provider", "level", "rating", "duration", "course_status", "course_url"]]], ignore_index=True)
            visible_columns = [column for column in ("skill", "skill_category", "course_name", "provider", "level", "rating", "duration", "course_status", "course_url") if column in available_courses]
            st.dataframe(available_courses[visible_columns], column_config={"course_url": st.column_config.LinkColumn("Course Link")}, use_container_width=True, hide_index=True)
        else:
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
            st.write(f"**Quiz: {selected_skill}** · {stage} · {priority} priority")
            if not dynamic:
                st.info("Log in with a registered account to start a personalized quiz attempt.")
            else:
                quiz_key = f"{user['id']}:{selected_skill}"
                if st.session_state.get("active_quiz_skill") != quiz_key:
                    st.session_state.active_quiz_skill = quiz_key
                    st.session_state.quiz_used_questions = st.session_state.get("quiz_used_questions", {})
                    st.session_state.quiz_used_questions.setdefault(quiz_key, set())
                    st.session_state.quiz_attempt = enrich_quiz_rows(selected_skill, auth_db.create_quiz_attempt(user["id"], selected_skill, stage, user["preferred_level"], st.session_state.quiz_used_questions[quiz_key]))
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_result = None
                    st.session_state.quiz_submitted = False
                quiz_rows = st.session_state.quiz_attempt
                total_questions = len(quiz_rows)
                if not quiz_rows:
                    st.warning("No quiz questions are available for this skill.")
                    st.stop()
                question_index = max(0, min(st.session_state.get("quiz_index", 0), total_questions - 1))
                st.session_state.quiz_index = question_index
                st.write(f"Question {question_index + 1} of {total_questions}")
                st.progress((question_index + 1) / total_questions, text=f"Question {question_index + 1} / {total_questions}")
                question_row = quiz_rows[question_index]
                options = [question_row["option_a"], question_row["option_b"], question_row["option_c"], question_row["option_d"]]
                answer_key = str(question_row["id"])
                chosen = st.radio(question_row["question"], options, index=options.index(st.session_state.quiz_answers[answer_key]) if answer_key in st.session_state.quiz_answers and st.session_state.quiz_answers[answer_key] in options else None, key=f"question_{quiz_key}_{question_row['id']}")
                if chosen is not None:
                    st.session_state.quiz_answers[answer_key] = chosen
                navigation = st.columns(3)
                with navigation[0]:
                    if st.button("Previous", disabled=question_index == 0):
                        st.session_state.quiz_index -= 1
                        st.rerun()
                with navigation[1]:
                    if st.button("Next", disabled=question_index >= total_questions - 1):
                        st.session_state.quiz_index += 1
                        st.rerun()
                with navigation[2]:
                    if st.button("Submit Quiz", disabled=len(st.session_state.quiz_answers) != total_questions, type="primary"):
                        st.session_state.quiz_result = auth_db.submit_quiz_attempt(user["id"], quiz_rows, st.session_state.quiz_answers)
                        auth_db.record_quiz_progress(user["id"], selected_skill, st.session_state.quiz_result[2])
                        st.session_state.quiz_submitted = True
                        st.rerun()
                if st.session_state.get("quiz_submitted", False) and st.session_state.get("quiz_result"):
                    score, total, percentage = st.session_state.quiz_result
                    st.subheader("Quiz Result")
                    st.write(f"**Skill:** {selected_skill}")
                    st.write(f"**Score:** {score} / {total}")
                    st.write(f"**Correct Answers:** {score}")
                    st.write(f"**Incorrect Answers:** {total - score}")
                    st.write(f"**Percentage:** {percentage:.2f}%")
                    st.write(f"**Status:** {'Passed' if percentage >= 70 else 'Failed'}")
                    feedback = "Excellent! Strong understanding." if percentage >= 90 else "Good job! Review the incorrect concepts." if percentage >= 70 else "Needs improvement. Practice this skill again." if percentage >= 50 else "Review the learning material and retry."
                    st.success(f"Score: {score} / {total}\n\nPercentage: {percentage:.1f}%\n\nCorrect Answers: {score}\n\nIncorrect Answers: {total - score}\n\n{feedback}")
                    incorrect_rows = [row for row in quiz_rows if str(st.session_state.quiz_answers.get(str(row["id"]), "")).strip().casefold() != str(row["correct_answer"]).strip().casefold()]
                    if incorrect_rows:
                        st.subheader("Topics to Review")
                        for row in incorrect_rows:
                            concept = row["concept"]
                            title, url = row["material_title"], row["material_url"]
                            st.markdown(f"- **{concept}** · [Learn Topic]({url})")
                    st.subheader("Answer Review")
                    for number, row in enumerate(quiz_rows, 1):
                        answer = st.session_state.quiz_answers.get(str(row["id"]), "Not answered")
                        correct = row["correct_answer"]
                        is_correct = str(answer).strip().casefold() == str(correct).strip().casefold()
                        concept = row["concept"]
                        invalid_concept = " ".join(("Core", "concept"))
                        if not str(concept).strip() or str(concept).strip().casefold() in {invalid_concept.casefold(), "unassigned"}:
                            st.error(f"Quiz data error: missing explicit concept for question {number}.")
                            st.stop()
                        explanation, title, url = row["explanation"], row["material_title"], row["material_url"]
                        with st.expander(f"{'✅' if is_correct else '❌'} Question {number} · {concept}"):
                            st.write("**Question:**", row["question"])
                            st.write("**Your Answer:**", answer)
                            st.write("**Correct Answer:**", correct)
                            st.write("**Result:**", "✅ Correct" if is_correct else "❌ Incorrect")
                            st.write("**Explanation:**", explanation)
                            if not is_correct:
                                st.write("**Why your answer is incorrect:** Your selected option does not match the concept tested.")
                                st.write("**Why the correct answer is correct:**", explanation)
                            st.write("**Topic:**", concept)
                            st.markdown(f"**Study Material:** [{title}]({url})")
                    if st.button("Retry quiz"):
                        st.session_state.quiz_used_questions[quiz_key].update(row["question"] for row in quiz_rows)
                        st.session_state.quiz_attempt = enrich_quiz_rows(selected_skill, auth_db.create_quiz_attempt(user["id"], selected_skill, stage, user["preferred_level"], st.session_state.quiz_used_questions[quiz_key]))
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_result = None
                        st.session_state.quiz_submitted = False
                        st.rerun()
            quiz_results = data.get("quiz_results", [])
            active_quiz = st.session_state.get("active_quiz_skill") == f"{user['id']}:{selected_skill}" if dynamic else False
            if quiz_results and not active_quiz:
                st.subheader("Quiz Results")
                results = pd.DataFrame(quiz_results)
                results["pass_fail"] = results["percentage"].map(lambda value: "Pass" if value >= 70 else "Fail")
                path_keys = path.get("skill_key", path.get("skill", pd.Series(dtype=str)))
                result_keys = results.get("skill_key", results.get("skill", pd.Series(dtype=str)))
                results["learning_path_status"] = result_keys.map(dict(zip(path_keys, path["progress_status"])))
                st.dataframe(results[["skill", "score", "total_questions", "percentage", "pass_fail", "submitted_at", "learning_path_status"]], use_container_width=True, hide_index=True)
            elif not active_quiz:
                st.info("No quiz attempts yet.")
    with rag_tab:
        question = st.text_input("Ask about your learning path", placeholder="What should I learn next?", key="rag_question")
        debug_rag = st.checkbox("Show retrieval diagnostics", value=False)
        if st.button("Ask / Generate Answer", type="primary"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                st.session_state.rag_results = rag_search(question.strip(), user)
                st.session_state.rag_answer = grounded_answer(question.strip(), st.session_state.rag_results, path, user)
        if st.session_state.get("rag_answer"):
            st.subheader("Answer")
            st.markdown(st.session_state.rag_answer)
            results = st.session_state.get("rag_results", [])
            if debug_rag and results and "error" not in results[0]:
                st.subheader("Retrieved Chunks")
                for result in results:
                    st.expander(f"{result['source']} · {result['score']:.3f}").write(result["content"])
                st.caption("Final context includes the retrieved chunks and the user's incomplete learning path.")


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
    dashboard(user)
