# AI Powered Personalised Learning Path Generator

## Problem statement
Students often receive generic course lists that do not reflect their current profile, career direction, skill gaps, pace, or preferred learning approach.

## Objective
Generate an ordered, personalised learning path from student data, career requirements, course availability, practice needs, and a retrieval-augmented knowledge base.

## Technologies
Python, pandas, Streamlit, FAISS, Sentence Transformers, NumPy, and CSV datasets.

## Architecture
Student input -> Agent 1 profile -> Agent 2 skill gap -> Agent 3 course recommendation -> Agent 4 ordered path -> Agent 5 quiz/practice -> Agent 6 final personalization -> FAISS retrieval -> Streamlit dashboard.

## Agents 1-6
- Agent 1 (`student_profile_agent.py`): combines student and intelligence data.
- Agent 2 (`skill_gap_agent.py`): compares explicit current skills with required career skills.
- Agent 3 (`course_recommendation_agent.py`): selects the highest-scoring available course for each gap.
- Agent 4 (`learning_path_agent.py`): orders skills from programming through professional skills.
- Agent 5 (`quiz_agent.py`): assigns deterministic quiz and practice recommendations.
- Agent 6 (`final_personalization_agent.py`): merges the ordered path and quiz plan.

## RAG implementation
`Dataset/Processed/RAG/retrieve.py` uses a FAISS index, `knowledge_chunks.pkl`, and the `all-MiniLM-L6-v2` embedding model. The dashboard loads these assets lazily when the RAG assistant is used, so the main dashboard can open from generated CSV files without re-running the pipeline.

## Data and outputs
Input datasets are under `Dataset/Processed/`, including student, career, employment-skills, and RAG knowledge-base data. Agent outputs are stored in `Dataset/Processed/RAG/`:
`student_profile_agent_output.csv`, `skill_gap_agent_output.csv`, `course_recommendation_agent_output.csv`, `learning_path_agent_output.csv`, `quiz_agent_output.csv`, and `final_personalized_learning_path.csv`.

## Personalization
Student ID selects the profile. Explicit current technical skills determine coverage; intelligence categories are not treated as technical skills. Required skills are staged, courses are selected by course score, no-course gaps remain visible, and practice intensity follows priority.

## Authentication and database
Registered learners use username/email login with salted PBKDF2 password hashes; plaintext passwords are never stored. Local SQLite uses `learning.db` and stores users, profiles, skill gaps, recommendations, learning paths, quizzes, quiz results, and progress. Every personalized query is scoped by the authenticated user ID. If an older local `learning_platform.db` exists and `learning.db` does not, startup copies it once without deleting either file, then applies safe schema migrations. Both database files are excluded from Git. Streamlit Cloud deployments must use a managed PostgreSQL/Supabase database for durable shared persistence; SQLite on Cloud is instance-local and can disappear after rebuilds.

## How to run
From the project root:

```powershell
py -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud deployment

The deployment entry point is `streamlit_app.py` at the repository root. It runs the same authenticated dashboard in `AI POWERED PERSONALISED LEARNING PATH GENERATOR/app.py`; it is not a demo or landing page.

Push `streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, the complete `AI POWERED PERSONALISED LEARNING PATH GENERATOR/` directory, and its tracked `Dataset/Processed/RAG/faiss_index.index` and `knowledge_chunks.pkl` assets. Do not push `.streamlit/secrets.toml`, `streamlit/secrets.toml`, `*.db`, or `*.sqlite`; use `.streamlit/secrets.toml.example` as the safe template.

In Streamlit Cloud, choose the repository, branch, and `streamlit_app.py` as the main file. Add the SMTP values from the example file in App settings > Secrets. Password reset also needs `APP_BASE_URL` set to the deployed app URL. The embedding model `all-MiniLM-L6-v2` is downloaded by Sentence Transformers on first RAG use and cached by the app.

`auth_db.py` is the current database layer used by the application. The current code intentionally keeps SQLite as the working local backend and warns when it is running with instance-local storage, rather than silently claiming that Cloud SQLite is durable. A managed PostgreSQL/Supabase adapter is the recommended next production step; this version rejects a non-SQLite setting instead of silently falling back to an ephemeral local file. Store the eventual connection URL in Streamlit Secrets, never in Git.

Deployment smoke test: register and log in, submit a profile, confirm Skill Analysis and Courses match by skill, update a Learning Path status, complete a quiz, verify the persisted Quiz Results row and progress count after rerun, then ask the RAG Assistant and enable retrieval diagnostics. Confirm a second account cannot see the first account's rows.

Open the local URL shown by Streamlit, select a Student ID, and explore the profile, gaps, courses, path, quizzes, and RAG assistant.

On first launch, create an account with a career goal, interests, current skills, and preferred level. Progress and quiz results are account-scoped. Streamlit Community Cloud's local filesystem is instance storage; use an external hosted database for multi-instance production persistence.

## Example workflow
For S1, select the student, review the Astronomer profile, inspect the 31-step path, open unavailable skills, and ask the RAG assistant what to learn first or why a skill is recommended.

## Limitations
The dashboard presents generated CSV outputs and does not retrain models. Current technical skills are marked unavailable when the source data does not explicitly provide them. RAG quality depends on the available knowledge chunks and embedding model.

## Future enhancements
Add authenticated student input, richer current-skill assessments, feedback-based progress tracking, quiz scoring, database storage, and scheduled pipeline refreshes.
