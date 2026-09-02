"""SQLite persistence and deterministic personalization for registered learners."""
import streamlit as st
from datetime import datetime, timezone
from email.message import EmailMessage
from contextlib import contextmanager
import hashlib
import hmac
import smtplib
import json
import os
import random
import secrets
import sqlite3
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "learning_platform.db"
def _smtp_config():
    """Read SMTP configuration from Streamlit secrets or environment variables."""

    try:
        secrets_config = st.secrets.get("smtp", {})
    except Exception:
        secrets_config = {}

    config = {
        "host": secrets_config.get(
            "host",
            os.getenv("SMTP_HOST", "smtp.gmail.com")
        ),
        "port": int(
            secrets_config.get(
                "port",
                os.getenv("SMTP_PORT", "587")
            )
        ),
        "username": secrets_config.get(
            "username",
            os.getenv("SMTP_USERNAME", "")
        ),
        "password": secrets_config.get(
            "password",
            os.getenv("SMTP_PASSWORD", "")
        ),
        "from_email": secrets_config.get(
            "from_email",
            os.getenv("SMTP_FROM_EMAIL", "")
        ),
    }

    if not config["username"] or not config["password"]:
        raise RuntimeError(
            "SMTP is not configured. Add SMTP credentials in Streamlit Cloud Secrets."
        )

    return config
STAGE_ORDER = ["Programming", "Data", "Statistics", "Machine Learning", "Databases and Big Data", "Cloud and Tools", "Professional Skills"]
STAGE_SKILLS = {
    "Programming": ["Python", "JavaScript", "React"],
    "Data": ["Data Integrity", "Data Preparation", "Data Mining", "Data Visualization", "Pandas"],
    "Statistics": ["Statistical Software", "Regressions", "Time Series Analysis", "Statistics"],
    "Machine Learning": ["Machine Learning", "Predictive Modeling", "Deep Learning", "Neural Networks", "Model Deployment", "Causal-Model Approaches", "Text Mining"],
    "Databases and Big Data": ["Sql", "Big Data", "Hadoop", "Apache Spark", "Hive", "Pig", "MongoDB"],
    "Cloud and Tools": ["Cloud Platforms", "Aws", "Docker", "Kubernetes", "Linux", "Networking", "Terraform", "Tableau", "TensorFlow", "Node.js"],
    "Professional Skills": ["Communication", "Stakeholder Engagement", "Strategic Thinking", "Problem Solving", "Project Management", "Teamwork"],
}
CAREER_REQUIREMENTS = {
    "data analyst": {"foundation": ["Python", "Sql"], "role": ["Pandas", "Data Preparation", "Statistics", "Data Visualization", "Tableau"]},
    "data scientist": {"foundation": ["Python", "Sql"], "role": ["Pandas", "Data Preparation", "Data Visualization", "Statistics", "Machine Learning", "Predictive Modeling", "Tableau"]},
    "machine learning engineer": {"foundation": ["Python"], "role": ["Pandas", "Statistics", "Machine Learning", "Deep Learning", "Neural Networks", "TensorFlow", "Model Deployment"]},
    "ml engineer": {"foundation": ["Python"], "role": ["Pandas", "Statistics", "Machine Learning", "Deep Learning", "Neural Networks", "TensorFlow", "Model Deployment"]},
    "web developer": {"foundation": ["JavaScript"], "role": ["React", "Node.js", "Communication", "Problem Solving"]},
    "cloud engineer": {"foundation": [], "role": ["Cloud Platforms", "Aws", "Docker", "Kubernetes", "Linux", "Networking", "Terraform"]},
}
CAREER_SKILLS = {name: requirements["foundation"] + requirements["role"] for name, requirements in CAREER_REQUIREMENTS.items()}

CURATED_COURSES = [
    ("python", "Python for Everybody", "Python", "Coursera", "University of Michigan", "Beginner", 4.8, 180000, "https://www.coursera.org/specializations/python", "Learn Python programming fundamentals.", "8 months", "None"),
    ("python-crash", "Python 3 Tutorial", "Python", "Official documentation", "Python Software Foundation", "Beginner", 5.0, 1, "https://docs.python.org/3/tutorial/", "The official Python language tutorial.", "Self-paced", "None"),
    ("pandas-docs", "Pandas User Guide", "Pandas", "Official documentation", "Pandas", "Intermediate", 5.0, 1, "https://pandas.pydata.org/docs/user_guide/", "Practical data manipulation with pandas.", "Self-paced", "Python"),
    ("data-analysis-python", "Data Analysis with Python", "Data Preparation", "Coursera", "IBM", "Intermediate", 4.7, 50000, "https://www.coursera.org/learn/data-analysis-with-python", "Analyze and visualize data using Python.", "5 weeks", "Python"),
    ("statistics-python", "Statistics with Python", "Statistics", "Coursera", "University of Michigan", "Intermediate", 4.7, 20000, "https://www.coursera.org/specializations/statistics-with-python", "Statistical analysis and inference with Python.", "4 months", "Python, Pandas"),
    ("sql-khan", "Intro to SQL", "Sql", "Khan Academy", "Khan Academy", "Beginner", 4.8, 10000, "https://www.khanacademy.org/computing/computer-programming/sql", "Query and analyze relational data.", "Self-paced", "None"),
    ("sql-coursera", "SQL for Data Science", "Sql", "Coursera", "University of California, Davis", "Beginner", 4.7, 30000, "https://www.coursera.org/learn/sql-for-data-science", "Use SQL to answer data science questions.", "4 weeks", "None"),
    ("visualization-tableau", "Fundamentals of Visualization with Tableau", "Data Visualization", "Coursera", "University of California, Davis", "Beginner", 4.6, 12000, "https://www.coursera.org/learn/communicating-data-insights", "Build clear visual data stories.", "4 weeks", "Data Preparation"),
    ("tableau-training", "Tableau Training", "Tableau", "Official documentation", "Salesforce", "Beginner", 5.0, 1, "https://www.tableau.com/learn/training", "Official Tableau learning paths.", "Self-paced", "Data Visualization"),
    ("ml-coursera", "Machine Learning", "Machine Learning", "Coursera", "Stanford University", "Intermediate", 4.9, 180000, "https://www.coursera.org/learn/machine-learning", "Foundations of machine learning.", "3 months", "Python, Statistics"),
    ("ml-google", "Machine Learning Crash Course", "Machine Learning", "Google", "Google", "Intermediate", 4.8, 1, "https://developers.google.com/machine-learning/crash-course", "Practical introduction to ML concepts.", "15 hours", "Python, Statistics"),
    ("deep-learning", "Deep Learning Specialization", "Deep Learning", "Coursera", "DeepLearning.AI", "Advanced", 4.9, 130000, "https://www.coursera.org/specializations/deep-learning", "Neural networks and deep learning.", "5 months", "Machine Learning"),
    ("tensorflow", "TensorFlow Core", "TensorFlow", "Official documentation", "Google", "Intermediate", 5.0, 1, "https://www.tensorflow.org/learn", "Official TensorFlow tutorials and guides.", "Self-paced", "Python, Machine Learning"),
    ("model-deployment", "Deploying Machine Learning Models", "Model Deployment", "Google Cloud Skills Boost", "Google Cloud", "Advanced", 4.7, 1, "https://www.cloudskillsboost.google/paths/17", "Deploy and operate ML workloads.", "Self-paced", "Machine Learning, Python"),
    ("aws-training", "AWS Skill Builder", "Aws", "AWS Skill Builder", "Amazon Web Services", "Beginner", 4.8, 1, "https://skillbuilder.aws/", "Official AWS training and learning plans.", "Self-paced", "None"),
    ("docker-training", "Docker getting started", "Docker", "Docker", "Docker", "Beginner", 5.0, 1, "https://docs.docker.com/get-started/", "Container fundamentals and Docker workflows.", "Self-paced", "None"),
    ("kubernetes-training", "Kubernetes Basics", "Kubernetes", "Kubernetes", "Cloud Native Computing Foundation", "Intermediate", 4.9, 1, "https://kubernetes.io/docs/tutorials/kubernetes-basics/", "Deploy and manage containerized applications.", "Self-paced", "Docker"),
]


@contextmanager
def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, career_goal TEXT NOT NULL, interests TEXT NOT NULL, current_skills TEXT NOT NULL, preferred_level TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS user_profiles (id INTEGER PRIMARY KEY, user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE, dominant_intelligence TEXT, intelligence_score REAL, profile_data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS skill_gaps (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, skill TEXT NOT NULL, skill_key TEXT NOT NULL DEFAULT '', skill_category TEXT NOT NULL DEFAULT 'Role-specific', current_level TEXT NOT NULL, required_level TEXT NOT NULL, gap TEXT NOT NULL, priority TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS course_recommendations (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, skill TEXT NOT NULL, skill_key TEXT NOT NULL DEFAULT '', course_id TEXT, course_name TEXT, institution TEXT, course_score REAL, course_status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS learning_path (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, sequence INTEGER NOT NULL, stage TEXT NOT NULL, skill TEXT NOT NULL, skill_key TEXT NOT NULL DEFAULT '', skill_category TEXT NOT NULL DEFAULT 'Role-specific', priority TEXT NOT NULL, course_name TEXT, course_status TEXT NOT NULL, learning_recommendation TEXT NOT NULL, progress_status TEXT NOT NULL DEFAULT 'Not Started');
        CREATE TABLE IF NOT EXISTS quizzes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, skill TEXT NOT NULL, skill_key TEXT NOT NULL DEFAULT '', stage TEXT NOT NULL, question TEXT NOT NULL, option_a TEXT NOT NULL, option_b TEXT NOT NULL, option_c TEXT NOT NULL, option_d TEXT NOT NULL, correct_answer TEXT NOT NULL, concept TEXT NOT NULL DEFAULT 'Unassigned', quiz_level TEXT NOT NULL DEFAULT 'Intermediate');
        CREATE TABLE IF NOT EXISTS quiz_results (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, quiz_id INTEGER NOT NULL REFERENCES quizzes(id), score INTEGER NOT NULL, total_questions INTEGER NOT NULL, percentage REAL NOT NULL, passed INTEGER NOT NULL DEFAULT 0, submitted_at TEXT NOT NULL, attempted_at TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, learning_step_id INTEGER NOT NULL REFERENCES learning_path(id), status TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, learning_step_id));
        CREATE TABLE IF NOT EXISTS quiz_answers (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE, user_answer TEXT NOT NULL, is_correct INTEGER NOT NULL DEFAULT 0, concept TEXT NOT NULL DEFAULT 'Unassigned', submitted_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS password_reset_tokens (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, token_hash TEXT NOT NULL UNIQUE, expires_at REAL NOT NULL, used_at TEXT);
        CREATE TABLE IF NOT EXISTS completed_courses (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, course_id TEXT NOT NULL, course_name TEXT NOT NULL, completed_at TEXT NOT NULL, UNIQUE(user_id, course_id));
        CREATE TABLE IF NOT EXISTS courses (course_id TEXT PRIMARY KEY, course_name TEXT NOT NULL, skill TEXT NOT NULL, topic TEXT NOT NULL, provider TEXT NOT NULL, institution TEXT NOT NULL, level TEXT NOT NULL, rating REAL, review_count INTEGER, course_url TEXT NOT NULL, description TEXT NOT NULL, duration TEXT NOT NULL, prerequisites TEXT NOT NULL);
        """)
        try:
            db.execute("ALTER TABLE quizzes ADD COLUMN quiz_level TEXT NOT NULL DEFAULT 'Intermediate'")
        except sqlite3.OperationalError:
            pass
        for statement in ("ALTER TABLE quizzes ADD COLUMN concept TEXT NOT NULL DEFAULT 'Unassigned'", "ALTER TABLE quiz_answers ADD COLUMN is_correct INTEGER NOT NULL DEFAULT 0", "ALTER TABLE quiz_answers ADD COLUMN concept TEXT NOT NULL DEFAULT 'Unassigned'"):
            try:
                db.execute(statement)
            except sqlite3.OperationalError:
                pass
        for statement in (
            "ALTER TABLE course_recommendations ADD COLUMN topic TEXT",
            "ALTER TABLE course_recommendations ADD COLUMN provider TEXT",
            "ALTER TABLE course_recommendations ADD COLUMN level TEXT",
            "ALTER TABLE course_recommendations ADD COLUMN rating REAL",
            "ALTER TABLE course_recommendations ADD COLUMN review_count INTEGER",
            "ALTER TABLE course_recommendations ADD COLUMN course_url TEXT",
            "ALTER TABLE course_recommendations ADD COLUMN duration TEXT",
            "ALTER TABLE course_recommendations ADD COLUMN prerequisites TEXT",
            "ALTER TABLE learning_path ADD COLUMN started_at TEXT",
            "ALTER TABLE learning_path ADD COLUMN completed_at TEXT",
            "ALTER TABLE learning_path ADD COLUMN course_id TEXT",
            "ALTER TABLE skill_gaps ADD COLUMN skill_key TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE skill_gaps ADD COLUMN skill_category TEXT NOT NULL DEFAULT 'Role-specific'",
            "ALTER TABLE course_recommendations ADD COLUMN skill_key TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_path ADD COLUMN skill_key TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_path ADD COLUMN skill_category TEXT NOT NULL DEFAULT 'Role-specific'",
            "ALTER TABLE quizzes ADD COLUMN skill_key TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE quiz_results ADD COLUMN passed INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE quiz_results ADD COLUMN attempted_at TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE progress ADD COLUMN started_at TEXT",
            "ALTER TABLE progress ADD COLUMN completed_at TEXT",
            "ALTER TABLE progress ADD COLUMN quiz_score INTEGER",
            "ALTER TABLE progress ADD COLUMN quiz_percentage REAL",
        ):
            try:
                db.execute(statement)
            except sqlite3.OperationalError:
                pass
        for table in ("skill_gaps", "course_recommendations", "learning_path", "quizzes"):
            db.execute(f"UPDATE {table} SET skill_key='' WHERE skill_key IS NULL")
            for row in db.execute(f"SELECT id, skill FROM {table} WHERE skill_key='' AND skill IS NOT NULL"):
                db.execute(f"UPDATE {table} SET skill_key=? WHERE id=?", (_norm(row["skill"]), row["id"]))
        db.execute("CREATE INDEX IF NOT EXISTS idx_skill_gaps_user ON skill_gaps(user_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_learning_path_user_sequence ON learning_path(user_id, sequence)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id)")
        db.execute("UPDATE quiz_results SET attempted_at=submitted_at WHERE attempted_at='' OR attempted_at IS NULL")
        db.execute("UPDATE quiz_results SET passed=CASE WHEN percentage >= 70 THEN 1 ELSE 0 END")


def _password_hash(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def _password_matches(password, stored):
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def register_user(full_name, username, email, password, career_goal, interests, current_skills, preferred_level):
    if not all(str(value).strip() for value in (full_name, username, email, password, career_goal, preferred_level)):
        raise ValueError("All required registration fields must be completed.")
    if password != password.strip() or len(password) < 8:
        raise ValueError("Password must be at least 8 characters and must not begin or end with spaces.")
    with connect() as db:
        try:
            cursor = db.execute("INSERT INTO users(full_name, username, email, password_hash, career_goal, interests, current_skills, preferred_level, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (full_name.strip(), username.strip(), email.strip().lower(), _password_hash(password), career_goal.strip(), interests.strip(), current_skills.strip(), preferred_level, datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError as error:
            raise ValueError("Username or email is already registered.") from error
        user_id = cursor.lastrowid
        db.execute("INSERT INTO user_profiles(user_id, profile_data) VALUES (?, ?)", (user_id, json.dumps({"interests": interests, "preferred_level": preferred_level})))
    return user_id


def authenticate(identifier, password):
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE lower(username)=lower(?) OR lower(email)=lower(?)", (identifier.strip(), identifier.strip())).fetchone()
    if row is None or not _password_matches(password, row["password_hash"]):
        return None
    return dict(row)


def request_password_reset(identifier, ttl_seconds=900):
    """Create a one-time password reset token and email it to the user."""
    print("PASSWORD RESET FUNCTION CALLED FOR:", identifier)
    identifier = str(identifier).strip()

    with connect() as db:
        row = db.execute(
            """
            SELECT id, email, full_name
            FROM users
            WHERE lower(username)=lower(?) OR lower(email)=lower(?)
            """,
            (identifier, identifier),
        ).fetchone()

        # Do not reveal whether an account exists.
        if row is None:
            return None

        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        expires_at = time.time() + ttl_seconds

        # Remove previous reset tokens for this user and expired tokens.
        db.execute(
            "DELETE FROM password_reset_tokens WHERE user_id=? OR expires_at<?",
            (row["id"], time.time()),
        )

        db.execute(
            """
            INSERT INTO password_reset_tokens(user_id, token_hash, expires_at)
            VALUES (?, ?, ?)
            """,
            (row["id"], token_hash, expires_at),
        )

    config = _smtp_config()

    message = EmailMessage()
    message["Subject"] = "Password Reset - AI Personalized Learning Path Generator"
    message["From"] = config["from_email"] or config["username"]
    message["To"] = row["email"]

    reset_link = (
        os.getenv(
            "APP_BASE_URL",
            "https://ai-powered-personalized-learning-path-generator-tzgwtbiahmtgth.streamlit.app/",
        ).rstrip("/")
        + "/?reset_token="
        + raw_token
    )

    message.set_content(
        f"""Hello {row["full_name"] or "there"},

We received a request to reset your password for the AI Personalized Learning Path Generator.

Use the following link to reset your password:

{reset_link}

This link expires in 15 minutes and can only be used once.

If you did not request a password reset, you can safely ignore this email.

Regards,
AI Personalized Learning Path Generator
"""
    )

    try:
        with smtplib.SMTP(config["host"], config["port"], timeout=20) as server:
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(message)
    except Exception as exc:
        # Remove the token if the email could not be sent.
        with connect() as db:
            db.execute(
                "DELETE FROM password_reset_tokens WHERE token_hash=?",
                (token_hash,),
            )
        raise RuntimeError(
            "Password reset email could not be sent. Please try again later."
        ) from exc

    # Never return the raw token to the application.
    return True

def reset_password(token, new_password):
    if new_password != str(new_password).strip() or len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters and must not begin or end with spaces.")
    token_hash = hashlib.sha256(str(token).encode()).hexdigest()
    with connect() as db:
        row = db.execute(
            "SELECT id, user_id FROM password_reset_tokens WHERE token_hash=? AND used_at IS NULL AND expires_at>?",
            (token_hash, time.time()),
        ).fetchone()
        if row is None:
            raise ValueError("This reset token is invalid or expired.")
        db.execute("UPDATE users SET password_hash=? WHERE id=?", (_password_hash(new_password), row["user_id"]))
        db.execute("UPDATE password_reset_tokens SET used_at=? WHERE id=?", (datetime.now(timezone.utc).isoformat(), row["id"]))
    return True


def get_user(user_id):
    with connect() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row is not None else None


def _norm(value):
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _requirements(career_goal, interests):
    goal = career_goal.lower()
    selected = next((skills for name, skills in CAREER_SKILLS.items() if name in goal), None)
    if selected is None:
        raise ValueError("Unsupported career goal. Choose a supported role such as Cloud Engineer, Data Scientist, ML Engineer, Data Analyst, or Web Developer.")
    extra = []
    result = []
    for skill in selected + extra:
        if _norm(skill) not in {_norm(item) for item in result}:
            result.append(skill)
    return result


def skill_category(career_goal, skill):
    goal = career_goal.lower()
    requirements = next((value for name, value in CAREER_REQUIREMENTS.items() if name in goal), None)
    if requirements is None:
        raise ValueError("Unsupported career goal.")
    return "Foundation" if _norm(skill) in {_norm(item) for item in requirements["foundation"]} else "Role-specific"


def _catalog_frame(ranked_courses):
    import pandas as pd
    rows = [{"course_id": course[0], "course_name": course[1], "required_skill": course[2], "topic": course[2], "provider": course[3], "institution": course[4], "level": course[5], "average_rating": course[6], "review_count": course[7], "course_url": course[8], "description": course[9], "duration": course[10], "prerequisites": course[11], "course_score": course[6] / 5 + min(course[7], 100000) / 1000000} for course in CURATED_COURSES]
    catalog = pd.DataFrame(rows)
    if not catalog.empty:
        catalog["skill_key"] = catalog["required_skill"].map(_norm)
    return catalog


def _completed_course_ids(db, user_id):
    return {row["course_id"] for row in db.execute("SELECT course_id FROM completed_courses WHERE user_id=?", (user_id,)).fetchall()}


def build_user_path(user, ranked_courses, completed_skills=None, completed_course_ids=None):
    required = _requirements(user["career_goal"], user["interests"])
    current = {_norm(item.strip()) for item in user["current_skills"].split(",") if item.strip()}
    current.update(_norm(item) for item in completed_skills or [])
    catalog = _catalog_frame(ranked_courses)
    completed_course_ids = completed_course_ids or set()
    skill_rows = []
    for skill in required:
        covered = _norm(skill) in current
        matches = catalog[catalog["skill_key"] == _norm(skill)].copy()
        matches = matches[~matches["course_id"].isin(completed_course_ids)]
        matches["course_score"] = __import__("pandas").to_numeric(matches["course_score"], errors="coerce")
        matches = matches.dropna(subset=["course_score"]).sort_values("course_score", ascending=False)
        course = matches.iloc[0] if not matches.empty else None
        if not covered:
            skill_rows.append((skill, covered, course))
    ordered = []
    for stage in STAGE_ORDER:
        for skill, covered, course in skill_rows:
            if _norm(skill) in {_norm(item) for item in STAGE_SKILLS[stage]}:
                ordered.append((stage, skill, covered, course))
    return ordered


def save_personalization(user_id, user, ranked_courses, recommendation):
    with connect() as db:
        catalog = _catalog_frame(ranked_courses)
        for _, course in catalog.iterrows():
            db.execute("INSERT OR REPLACE INTO courses(course_id, course_name, skill, topic, provider, institution, level, rating, review_count, course_url, description, duration, prerequisites) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(course.get(column, "") for column in ("course_id", "course_name", "required_skill", "topic", "provider", "institution", "level", "average_rating", "review_count", "course_url", "description", "duration", "prerequisites")))
        db.execute("DELETE FROM quiz_answers WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM quiz_results WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM progress WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM quizzes WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM skill_gaps WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM course_recommendations WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM learning_path WHERE user_id=?", (user_id,))
        for sequence, (stage, skill, covered, course) in enumerate(recommendation, 1):
            priority = "High" if not covered else "Low"
            gap = "Covered" if covered else "Gap"
            skill_key = _norm(skill)
            category = skill_category(user["career_goal"], skill)
            db.execute("INSERT INTO skill_gaps(user_id, skill, skill_key, skill_category, current_level, required_level, gap, priority) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, skill, skill_key, category, "Available" if covered else "Not Available", "Required", gap, priority))
            status = "Course Available" if course is not None and not covered else "No Course Available" if not covered else "Covered"
            if course is not None and not covered:
                db.execute("INSERT INTO course_recommendations(user_id, skill, skill_key, course_id, course_name, institution, course_score, course_status, topic, provider, level, rating, review_count, course_url, duration, prerequisites) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (user_id, skill, skill_key, course["course_id"], course["course_name"], course["institution"], float(course["course_score"]), status, course["topic"], course["provider"], course["level"], float(course["average_rating"]), int(course["review_count"]), course["course_url"], course["duration"], course["prerequisites"]))
            recommendation_text = "Use examples, categorization, comparisons and pattern recognition." if user["preferred_level"] == "Beginner" else "Apply the concept through progressively challenging projects."
            db.execute("INSERT INTO learning_path(user_id, sequence, stage, skill, skill_key, skill_category, priority, course_id, course_name, course_status, learning_recommendation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (user_id, sequence, stage, skill, skill_key, category, priority, course["course_id"] if course is not None and not covered else "", course["course_name"] if course is not None and not covered else "", status, recommendation_text))


def user_data(user_id):
    with connect() as db:
        data = {table: [dict(row) for row in db.execute(f"SELECT * FROM {table} WHERE user_id=? ORDER BY id", (user_id,))] for table in ("skill_gaps", "course_recommendations", "learning_path", "quizzes", "quiz_answers", "progress")}
        data["quiz_results"] = [dict(row) for row in db.execute("SELECT quiz_results.*, quizzes.skill, quizzes.skill_key FROM quiz_results JOIN quizzes ON quizzes.id=quiz_results.quiz_id AND quizzes.user_id=quiz_results.user_id WHERE quiz_results.user_id=? ORDER BY quiz_results.id", (user_id,))]
        return data


def update_progress(user_id, step_id, status):
    with connect() as db:
        now = datetime.now(timezone.utc).isoformat()
        step = db.execute("SELECT course_id, course_name, progress_status FROM learning_path WHERE id=? AND user_id=?", (step_id, user_id)).fetchone()
        if step is None:
            raise KeyError("Learning step does not belong to this account.")
        passed_quiz = db.execute("SELECT 1 FROM progress WHERE user_id=? AND learning_step_id=? AND quiz_percentage>=70", (user_id, step_id)).fetchone()
        if passed_quiz:
            status = "Completed"
        started_at = now if status in ("In Progress", "Completed") else None
        completed_at = now if status == "Completed" else None
        db.execute("INSERT INTO progress(user_id, learning_step_id, status, updated_at, started_at, completed_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, learning_step_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, started_at=COALESCE(progress.started_at, excluded.started_at), completed_at=excluded.completed_at", (user_id, step_id, status, now, started_at, completed_at))
        db.execute("UPDATE learning_path SET progress_status=?, started_at=COALESCE(started_at, ?), completed_at=? WHERE id=? AND user_id=?", (status, started_at, completed_at, step_id, user_id))
        if status == "Completed" and step["course_id"]:
            db.execute("INSERT OR IGNORE INTO completed_courses(user_id, course_id, course_name, completed_at) VALUES (?, ?, ?, ?)", (user_id, step["course_id"], step["course_name"], now))
            db.execute("UPDATE course_recommendations SET course_status='Completed' WHERE user_id=? AND course_id=?", (user_id, step["course_id"]))


def record_quiz_progress(user_id, skill, percentage):
    """Update only the logged-in user's learning step matching the quiz skill."""
    status = "Completed" if percentage >= 70 else "In Progress"
    with connect() as db:
        step = db.execute(
            "SELECT id, progress_status FROM learning_path WHERE user_id=? AND skill_key=? ORDER BY sequence LIMIT 1",
            (user_id, _norm(skill)),
        ).fetchone()
        if step is None:
            raise KeyError(f"No learning-path step found for skill '{skill}'.")
        if step["progress_status"] == "Completed" and status != "Completed":
            status = "Completed"
        latest_result = db.execute("SELECT score, total_questions FROM quiz_results WHERE user_id=? AND quiz_id IN (SELECT id FROM quizzes WHERE user_id=? AND skill_key=?) ORDER BY id DESC LIMIT 1", (user_id, user_id, _norm(skill))).fetchone()
        score = latest_result["score"] if latest_result else round(percentage / 100 * 7)
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO progress(user_id, learning_step_id, status, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, learning_step_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, quiz_score=excluded.quiz_score, quiz_percentage=excluded.quiz_percentage",
            (user_id, step["id"], status, now),
        )
        db.execute("UPDATE progress SET quiz_score=?, quiz_percentage=?, started_at=COALESCE(started_at, ?), completed_at=? WHERE user_id=? AND learning_step_id=?", (score, percentage, now, now if status == "Completed" else None, user_id, step["id"]))
        db.execute(
            "UPDATE learning_path SET progress_status=?, started_at=COALESCE(started_at, ?), completed_at=? WHERE id=? AND user_id=?",
            (status, now, now if status == "Completed" else None, step["id"], user_id),
        )
    return status


def save_quiz_result(user_id, quiz_id, score, total_questions):
    percentage = score / total_questions * 100
    with connect() as db: db.execute("INSERT INTO quiz_results(user_id, quiz_id, score, total_questions, percentage, submitted_at) VALUES (?, ?, ?, ?, ?, ?)", (user_id, quiz_id, score, total_questions, percentage, datetime.now(timezone.utc).isoformat()))
    return percentage


QUIZ_BANK = {
    "Python": [
        ("Which keyword defines a function?", ["func", "def", "lambda", "method"], "def"),
        ("Which type is mutable and ordered?", ["Tuple", "List", "Set", "String"], "List"),
        ("Which structure stores key-value pairs?", ["List", "Tuple", "Dictionary", "Set"], "Dictionary"),
        ("Which statement handles an exception?", ["try/except", "if/else", "for/in", "with/as"], "try/except"),
        ("What does len() return?", ["An object address", "The number of items", "A data type", "A loop"], "The number of items"),
        ("Which loop iterates over items in a sequence?", ["for", "switch", "repeat", "iterate"], "for"),
        ("What does a class define?", ["A blueprint for objects", "Only a loop", "A package installer", "A comment"], "A blueprint for objects"),
        ("Which symbol starts a comment?", ["//", "#", "<!--", "**"], "#"),
        ("What is None?", ["The absence of a value", "A number", "A loop", "A module"], "The absence of a value"),
        ("Which keyword imports a module?", ["include", "using", "import", "require"], "import"),
    ],
    "Sql": [
        ("Which clause filters rows?", ["WHERE", "ORDER BY", "JOIN", "GROUP BY"], "WHERE"),
        ("Which command reads rows?", ["SELECT", "INSERT", "UPDATE", "DELETE"], "SELECT"),
        ("Which keyword combines related tables?", ["JOIN", "LINK", "MERGE", "ATTACH"], "JOIN"),
        ("Which function counts rows?", ["SUM", "COUNT", "TOTAL", "ROWS"], "COUNT"),
        ("Which clause groups rows for aggregation?", ["GROUP BY", "ORDER BY", "WHERE", "HAVING"], "GROUP BY"),
        ("Which clause filters grouped results?", ["HAVING", "WHERE", "FILTER", "GROUP"], "HAVING"),
        ("What uniquely identifies a row?", ["Primary key", "View", "Alias", "Index page"], "Primary key"),
        ("Which command adds a new row?", ["INSERT", "CREATE", "ALTER", "APPEND"], "INSERT"),
        ("Which function calculates an average?", ["MEAN", "AVG", "AVERAGE", "MID"], "AVG"),
        ("What is a subquery?", ["A query inside another query", "A deleted table", "A database user", "A column type"], "A query inside another query"),
    ],
    "Pandas": [
        ("What is the main difference between a Series and a DataFrame?", ["A Series is one-dimensional; a DataFrame is two-dimensional", "A Series is always text", "A DataFrame cannot have labels", "They are identical"], "A Series is one-dimensional; a DataFrame is two-dimensional"),
        ("Which accessor selects rows and columns by labels?", ["iloc", "loc", "at only", "axis"], "loc"),
        ("Which expression filters rows where age is greater than 18?", ["df[df['age'] > 18]", "df.filter(age > 18)", "df.rows(18)", "df.where('age')"], "df[df['age'] > 18]"),
        ("What does groupby commonly enable?", ["Split-apply-combine analysis", "Password hashing", "File compression", "Plot rendering only"], "Split-apply-combine analysis"),
        ("Which operation combines DataFrames using matching keys?", ["merge", "sort_values", "describe", "astype"], "merge"),
        ("Which method replaces missing values?", ["fillna", "dropindex", "replaceall", "missing"], "fillna"),
        ("Which method orders rows by a column?", ["sort_values", "order_rows", "arrange", "rank_table"], "sort_values"),
        ("Which method creates a spreadsheet-style summary?", ["pivot_table", "reshape_only", "table_view", "summarize_file"], "pivot_table"),
        ("Which function applies a Python function element-wise to a Series?", ["map", "merge", "concat", "join"], "map"),
        ("Which function reads a CSV file into a DataFrame?", ["read_csv", "load_csv_data", "open_table", "csv_import"], "read_csv"),
    ],
    "Machine Learning": [
        ("What does supervised learning use?", ["Labeled data", "No data", "Only rules", "Passwords"], "Labeled data"),
        ("Which task predicts a continuous value?", ["Regression", "Classification", "Clustering", "Sorting"], "Regression"),
        ("Which task predicts a category?", ["Classification", "Regression", "Sampling", "Aggregation"], "Classification"),
        ("Why split train and test data?", ["Evaluate generalization", "Increase file size", "Remove labels", "Avoid features"], "Evaluate generalization"),
        ("What is overfitting?", ["Memorizing training data", "Having no data", "Using a test set", "Scaling features"], "Memorizing training data"),
        ("Which metric is common for classification?", ["Accuracy", "Mean only", "Variance only", "File size"], "Accuracy"),
        ("What does a feature represent?", ["An input variable", "A password", "A prediction only", "A database server"], "An input variable"),
        ("What does clustering do?", ["Groups similar items", "Predicts labels only", "Deletes outliers", "Sorts columns"], "Groups similar items"),
        ("Why scale numeric features?", ["Put values on comparable ranges", "Remove the target", "Create labels", "Encrypt data"], "Put values on comparable ranges"),
        ("What is a model's prediction?", ["Its output for input data", "Its training file", "A feature name", "A database key"], "Its output for input data"),
    ],
}


CONCEPT_KEYWORDS = {
    "Python": [("Variables and data types", ("type", "none")), ("Collections", ("mutable", "dictionary")), ("Functions", ("function",)), ("Exception handling", ("exception",)), ("Iteration", ("loop",)), ("Object-oriented programming", ("class",)), ("Modules", ("module",)), ("Comments", ("comment",))],
    "Pandas": [("Series and DataFrame", ("series", "dataframe")), ("Indexing", ("accessor",)), ("Filtering", ("filter",)), ("GroupBy", ("groupby",)), ("Merge and join", ("merge",)), ("Missing values", ("missing",)), ("Sorting", ("sort",)), ("Pivot tables", ("pivot",)), ("Apply and map", ("function",)), ("Reading CSV", ("csv",))],
    "Sql": [("WHERE filtering", ("filters",)), ("SELECT", ("reads",)), ("JOINs", ("combines",)), ("Aggregate functions", ("counts", "average")), ("GROUP BY", ("groups",)), ("HAVING", ("grouped",)), ("Primary keys", ("identifies",)), ("INSERT", ("new row",)), ("Subqueries", ("subquery",))],
    "Machine Learning": [("Supervised learning", ("supervised",)), ("Regression", ("continuous",)), ("Classification", ("category",)), ("Train/test split", ("split",)), ("Overfitting", ("overfitting",)), ("Evaluation metrics", ("metric",)), ("Features", ("feature",)), ("Clustering", ("clustering",)), ("Feature scaling", ("scale",)), ("Predictions", ("prediction",))],
}

QUIZ_CONCEPTS = {
    "Python": ["Variables and data types", "Collections", "Dictionaries", "Exception handling", "Built-in functions", "Iteration", "Object-oriented programming", "Comments", "Special values", "Modules"],
    "Pandas": ["Series and DataFrame", "Indexing", "Filtering", "GroupBy", "Merge and join", "Missing values", "Sorting", "Pivot tables", "Apply and map", "Reading CSV"],
    "Sql": ["WHERE filtering", "SELECT", "JOINs", "Aggregate functions", "GROUP BY", "HAVING", "Primary keys", "INSERT", "AVG function", "Subqueries"],
    "Machine Learning": ["Supervised learning", "Regression", "Classification", "Train/test split", "Overfitting", "Evaluation metrics", "Features", "Clustering", "Feature scaling", "Predictions"],
}


def question_concept(skill, question):
    bank = next((questions for name, questions in QUIZ_BANK.items() if _norm(name) == _norm(skill)), [])
    concepts = next((concepts for name, concepts in QUIZ_CONCEPTS.items() if _norm(name) == _norm(skill)), [])
    for index, item in enumerate(bank):
        if item[0] == question and index < len(concepts):
            return concepts[index]
    raise ValueError(f"No explicit concept metadata exists for quiz question: {question}")


def _quiz_count(preferred_level):
    return {"Beginner": 5, "Intermediate": 7, "Advanced": 10}.get(preferred_level, 7)


def create_quiz_attempt(user_id, skill, stage, preferred_level, excluded_questions=None):
    """Create a fresh randomized, skill-specific quiz attempt for one user."""
    bank = next((questions for name, questions in QUIZ_BANK.items() if _norm(name) == _norm(skill)), [])
    questions = list(bank)
    generated_fallback = not questions
    if not questions:
        concepts = ["foundations", "workflow", "inputs", "outputs", "validation", "application", "troubleshooting", "comparison", "best practice", "interpretation"]
        questions = [(f"{skill} concept check: which action best supports {concept}?", [f"Apply {skill} through {concept}", "Skip the topic", "Ignore feedback", "Delete the notes"], f"Apply {skill} through {concept}") for concept in concepts]
    required_count = _quiz_count(preferred_level)
    excluded_questions = set(excluded_questions or [])
    unused_questions = [question for question in questions if question[0] not in excluded_questions]
    questions = unused_questions + [question for question in questions if question[0] in excluded_questions]
    randomizer = random.SystemRandom()
    randomizer.shuffle(questions)
    required_question = next((question for question in questions if question[0] == "Which keyword imports a module?"), None)
    if _norm(skill) == "python" and required_question is not None:
        questions.remove(required_question)
        questions.insert(0, required_question)
    with connect() as db:
        rows = []
        for question, options, answer in questions[:required_count]:
            shuffled = list(options); randomizer.shuffle(shuffled)
            concept = question_concept(skill, question) if not generated_fallback else f"{skill} {question.split('supports progress in ')[-1].rstrip('?').title()}"
            cursor = db.execute("INSERT INTO quizzes(user_id, skill, skill_key, stage, question, option_a, option_b, option_c, option_d, correct_answer, quiz_level, concept) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (user_id, skill, _norm(skill), stage, question, *shuffled, answer, preferred_level, concept))
            rows.append(dict(db.execute("SELECT * FROM quizzes WHERE id=?", (cursor.lastrowid,)).fetchone()))
    return rows


def submit_quiz_attempt(user_id, quiz_rows, answers):
    """Persist answers and one immutable result for a quiz attempt."""
    if not quiz_rows:
        raise ValueError("This quiz has no questions.")
    score = 0
    with connect() as db:
        question_ids = [row["id"] for row in quiz_rows]
        placeholders = ",".join("?" for _ in question_ids)
        owned = db.execute(f"SELECT id FROM quizzes WHERE user_id=? AND id IN ({placeholders})", [user_id, *question_ids]).fetchall()
        if len(owned) != len(question_ids):
            raise ValueError("Quiz questions do not belong to this account.")
        for row in quiz_rows:
            answer = answers.get(str(row["id"]), "")
            is_correct = str(answer).strip().casefold() == str(row["correct_answer"]).strip().casefold()
            if is_correct: score += 1
            db.execute("INSERT INTO quiz_answers(user_id, quiz_id, user_answer, is_correct, concept, submitted_at) VALUES (?, ?, ?, ?, ?, ?)", (user_id, row["id"], answer, int(is_correct), row["concept"], datetime.now(timezone.utc).isoformat()))
        percentage = score / len(quiz_rows) * 100
        submitted_at = datetime.now(timezone.utc).isoformat()
        db.execute("INSERT INTO quiz_results(user_id, quiz_id, score, total_questions, percentage, passed, submitted_at, attempted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, quiz_rows[0]["id"], score, len(quiz_rows), percentage, int(percentage >= 70), submitted_at, submitted_at))
    return score, len(quiz_rows), percentage
