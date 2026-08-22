"""SQLite persistence and deterministic personalization for registered learners."""

from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "learning_platform.db"
STAGE_ORDER = ["Programming", "Data", "Statistics", "Machine Learning", "Databases and Big Data", "Cloud and Tools", "Professional Skills"]
STAGE_SKILLS = {
    "Programming": ["Python", "JavaScript", "React"],
    "Data": ["Data Integrity", "Data Preparation", "Data Mining", "Data Visualization", "Pandas"],
    "Statistics": ["Statistical Software", "Regressions", "Time Series Analysis", "Statistics"],
    "Machine Learning": ["Machine Learning", "Predictive Modeling", "Causal-Model Approaches", "Text Mining"],
    "Databases and Big Data": ["Sql", "Big Data", "Hadoop", "Apache Spark", "Hive", "Pig", "MongoDB"],
    "Cloud and Tools": ["Cloud Platforms", "Aws", "Tableau", "Docker", "Kubernetes", "Node.js"],
    "Professional Skills": ["Communication", "Stakeholder Engagement", "Strategic Thinking", "Problem Solving", "Project Management", "Teamwork"],
}
CAREER_SKILLS = {
    "data scientist": ["Python", "Sql", "Pandas", "Data Preparation", "Data Visualization", "Statistics", "Machine Learning", "Predictive Modeling", "Tableau"],
    "web developer": ["JavaScript", "Python", "Data Preparation", "Sql", "Node.js", "Communication", "Problem Solving"],
    "cloud engineer": ["Python", "Cloud Platforms", "Aws", "Docker", "Kubernetes", "Sql", "Big Data", "Communication"],
}


def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, career_goal TEXT NOT NULL, interests TEXT NOT NULL, current_skills TEXT NOT NULL, preferred_level TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS user_profiles (id INTEGER PRIMARY KEY, user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE, dominant_intelligence TEXT, intelligence_score REAL, profile_data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS skill_gaps (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, skill TEXT NOT NULL, current_level TEXT NOT NULL, required_level TEXT NOT NULL, gap TEXT NOT NULL, priority TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS course_recommendations (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, skill TEXT NOT NULL, course_id TEXT, course_name TEXT, institution TEXT, course_score REAL, course_status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS learning_path (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, sequence INTEGER NOT NULL, stage TEXT NOT NULL, skill TEXT NOT NULL, priority TEXT NOT NULL, course_name TEXT, course_status TEXT NOT NULL, learning_recommendation TEXT NOT NULL, progress_status TEXT NOT NULL DEFAULT 'Not Started');
        CREATE TABLE IF NOT EXISTS quizzes (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, skill TEXT NOT NULL, stage TEXT NOT NULL, question TEXT NOT NULL, option_a TEXT NOT NULL, option_b TEXT NOT NULL, option_c TEXT NOT NULL, option_d TEXT NOT NULL, correct_answer TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS quiz_results (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, quiz_id INTEGER NOT NULL REFERENCES quizzes(id), score INTEGER NOT NULL, total_questions INTEGER NOT NULL, percentage REAL NOT NULL, submitted_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, learning_step_id INTEGER NOT NULL REFERENCES learning_path(id), status TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(user_id, learning_step_id));
        """)


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


def _norm(value):
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _requirements(career_goal, interests):
    goal = career_goal.lower()
    selected = next((skills for name, skills in CAREER_SKILLS.items() if name in goal), CAREER_SKILLS["data scientist"])
    extra = [item.strip() for item in interests.split(",") if item.strip()]
    result = []
    for skill in selected + extra:
        if _norm(skill) not in {_norm(item) for item in result}:
            result.append(skill)
    return result


def build_user_path(user, ranked_courses):
    required = _requirements(user["career_goal"], user["interests"])
    current = {_norm(item.strip()) for item in user["current_skills"].split(",") if item.strip()}
    skill_rows = []
    for skill in required:
        covered = _norm(skill) in current
        matches = ranked_courses[ranked_courses["required_skill"].map(_norm) == _norm(skill)].copy()
        matches["course_score"] = __import__("pandas").to_numeric(matches["course_score"], errors="coerce")
        matches = matches.dropna(subset=["course_score"]).sort_values("course_score", ascending=False)
        course = matches.iloc[0] if not matches.empty else None
        skill_rows.append((skill, covered, course))
    ordered = []
    for stage in STAGE_ORDER:
        for skill, covered, course in skill_rows:
            if not covered and _norm(skill) in {_norm(item) for item in STAGE_SKILLS[stage]}:
                ordered.append((stage, skill, covered, course))
    return ordered


def save_personalization(user_id, user, ranked_courses, recommendation):
    with connect() as db:
        db.execute("DELETE FROM skill_gaps WHERE user_id=?", (user_id,)); db.execute("DELETE FROM course_recommendations WHERE user_id=?", (user_id,)); db.execute("DELETE FROM learning_path WHERE user_id=?", (user_id,)); db.execute("DELETE FROM quizzes WHERE user_id=?", (user_id,))
        for sequence, (stage, skill, covered, course) in enumerate(recommendation, 1):
            priority = "High" if not covered else "Low"
            gap = "Covered" if covered else "Gap"
            db.execute("INSERT INTO skill_gaps(user_id, skill, current_level, required_level, gap, priority) VALUES (?, ?, ?, ?, ?, ?)", (user_id, skill, "Available" if covered else "Not Available", "Required", gap, priority))
            status = "Course Available" if course is not None and not covered else "No Course Available" if not covered else "Covered"
            if course is not None and not covered:
                db.execute("INSERT INTO course_recommendations(user_id, skill, course_id, course_name, institution, course_score, course_status) VALUES (?, ?, ?, ?, ?, ?, ?)", (user_id, skill, course["course_id"], course["course_name"], course["institution"], float(course["course_score"]), status))
            recommendation_text = "Use examples, categorization, comparisons and pattern recognition." if user["preferred_level"] == "Beginner" else "Apply the concept through progressively challenging projects."
            db.execute("INSERT INTO learning_path(user_id, sequence, stage, skill, priority, course_name, course_status, learning_recommendation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, sequence, stage, skill, priority, course["course_name"] if course is not None and not covered else "", status, recommendation_text))
            if not covered:
                questions = [(f"Which action supports progress in {skill}?", "Practice the skill", "Skip practice", "Ignore feedback", "Delete notes", "Practice the skill"), (f"What is a useful way to learn {skill}?", "Apply it to a small problem", "Avoid examples", "Ignore results", "Use no resources", "Apply it to a small problem")]
                for question, option_a, option_b, option_c, option_d, answer in questions:
                    db.execute("INSERT INTO quizzes(user_id, skill, stage, question, option_a, option_b, option_c, option_d, correct_answer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (user_id, skill, stage, question, option_a, option_b, option_c, option_d, answer))


def user_data(user_id):
    with connect() as db:
        return {table: [dict(row) for row in db.execute(f"SELECT * FROM {table} WHERE user_id=? ORDER BY id", (user_id,))] for table in ("skill_gaps", "course_recommendations", "learning_path", "quizzes", "quiz_results", "progress")}


def update_progress(user_id, step_id, status):
    with connect() as db:
        db.execute("INSERT INTO progress(user_id, learning_step_id, status, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, learning_step_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at", (user_id, step_id, status, datetime.now(timezone.utc).isoformat()))
        db.execute("UPDATE learning_path SET progress_status=? WHERE id=? AND user_id=?", (status, step_id, user_id))


def save_quiz_result(user_id, quiz_id, score, total_questions):
    percentage = score / total_questions * 100
    with connect() as db: db.execute("INSERT INTO quiz_results(user_id, quiz_id, score, total_questions, percentage, submitted_at) VALUES (?, ?, ?, ?, ?, ?)", (user_id, quiz_id, score, total_questions, percentage, datetime.now(timezone.utc).isoformat()))
    return percentage
