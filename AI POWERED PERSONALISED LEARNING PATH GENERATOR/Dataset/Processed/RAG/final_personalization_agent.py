"""Agent 6: Combine the ordered learning path with quiz recommendations."""

import argparse
from pathlib import Path

import pandas as pd


INPUT_FILES = {
    "profile": "student_profile_agent_output.csv",
    "skill_gap": "skill_gap_agent_output.csv",
    "learning_path": "learning_path_agent_output.csv",
    "quiz": "quiz_agent_output.csv",
    "recommendations": "course_recommendation_agent_output.csv",
}
OUTPUT_PATH = "final_personalized_learning_path.csv"
MERGE_KEYS = ["Student ID", "Sequence", "Skill"]
BASE_COLUMNS = [
    "Student ID", "Sequence", "Stage", "Skill", "Priority", "Course ID",
    "Course Name", "Institution", "Average Rating", "Review Count",
    "Course Score", "Course Status", "Learning Recommendation",
]
QUIZ_COLUMNS = ["Quiz Type", "Number of Questions", "Practice Level", "Quiz Status"]


REQUIRED_COLUMNS = {
    "profile": ["Student ID"],
    "skill_gap": ["Student ID", "Skill", "Priority"],
    "learning_path": BASE_COLUMNS,
    "quiz": MERGE_KEYS + QUIZ_COLUMNS,
    "recommendations": ["Student ID", "Skill", "Course Status"],
}


def _require_columns(dataframe, columns, filename):
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{filename} is missing required column(s): {', '.join(missing)}")


def _student_ids(dataframe, filename):
    values = dataframe["Student ID"].dropna().astype(str).str.strip()
    if values.empty or (values == "").any():
        raise ValueError(f"{filename} contains missing or invalid Student ID values.")
    return set(values)


def _reject_duplicates(dataframe, keys, filename):
    duplicate_rows = dataframe[dataframe.duplicated(keys, keep=False)]
    if not duplicate_rows.empty:
        raise ValueError(f"{filename} contains duplicate records for keys: {', '.join(keys)}")


def load_inputs(input_folder="."):
    """Load and validate all five upstream agent outputs."""
    folder = Path(input_folder)
    data = {}
    for name, filename in INPUT_FILES.items():
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
        data[name] = pd.read_csv(path)
        _require_columns(data[name], REQUIRED_COLUMNS[name], filename)

    id_sets = {name: _student_ids(frame, INPUT_FILES[name]) for name, frame in data.items()}
    all_ids = set().union(*id_sets.values())
    if len(all_ids) != 1 or any(ids != all_ids for ids in id_sets.values()):
        details = "; ".join(f"{name}: {sorted(ids)}" for name, ids in id_sets.items())
        raise ValueError(f"Input files do not belong to one Student ID ({details}).")

    _reject_duplicates(data["learning_path"], MERGE_KEYS, INPUT_FILES["learning_path"])
    _reject_duplicates(data["quiz"], MERGE_KEYS, INPUT_FILES["quiz"])
    return data


def build_final_path(data):
    """Merge quiz fields onto the Agent 4 path without changing its order."""
    learning_path = data["learning_path"].copy()
    quiz = data["quiz"][MERGE_KEYS + QUIZ_COLUMNS].copy()
    merged = learning_path.merge(quiz, on=MERGE_KEYS, how="left", sort=False, validate="one_to_one")
    if len(merged) != len(learning_path):
        raise ValueError("Quiz merge changed the number of learning-path records.")

    missing_quiz = merged[QUIZ_COLUMNS].isna().any(axis=1)
    if missing_quiz.any():
        missing_keys = merged.loc[missing_quiz, MERGE_KEYS].to_dict("records")
        raise ValueError(f"Quiz output is missing learning-path records: {missing_keys}")
    return merged[BASE_COLUMNS + QUIZ_COLUMNS]


def save_final_path(final_path, output_path=OUTPUT_PATH):
    final_path.to_csv(output_path, index=False)
    return output_path


def print_summary(final_path):
    available = final_path["Course Status"] == "Course Available"
    gaps = final_path["Course Status"] == "Skill Gap - No Course"
    high = final_path["Priority"].astype(str).str.casefold() == "high"
    medium = final_path["Priority"].astype(str).str.casefold() == "medium"
    quizzes = final_path["Quiz Status"] == "Quiz Recommended"
    practice = final_path["Quiz Status"] == "Practice Required - No Course"
    print("FINAL PERSONALIZATION AGENT")
    print("===========================")
    print("Student ID:", final_path["Student ID"].iloc[0])
    print("Total learning steps:", len(final_path))
    print("Steps with courses:", available.sum())
    print("Skill gaps:", gaps.sum())
    print("High-priority steps:", high.sum())
    print("Medium-priority steps:", medium.sum())
    print("Quiz recommendations:", quizzes.sum())
    print("Practice-required steps:", practice.sum())
    print("\nFinal personalized learning path")
    print(final_path.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Build the final personalized learning path.")
    parser.add_argument("--input-folder", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    try:
        final_path = build_final_path(load_inputs(args.input_folder))
        print_summary(final_path)
        print("\nFinal output saved to:", save_final_path(final_path, args.output))
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
