"""Agent 2: Compare a student's explicit skills with career requirements."""

import argparse
from pathlib import Path

import pandas as pd


STUDENT_ID = "S1"
AGENT_1_OUTPUT_PATH = "student_profile_agent_output.csv"
STUDENT_DATA_PATH = r"..\student\student_dataset_cleaned.csv"
REQUIRED_SKILLS_PATH = "skill_gap_analysis.csv"
FALLBACK_SKILLS_PATH = "skill_gap_analysis.csv"
OUTPUT_PATH = "skill_gap_agent_output.csv"
STUDENT_ID_COLUMN = "Student_ID"

OUTPUT_COLUMNS = [
    "Student ID",
    "Skill",
    "Current Status/Level",
    "Required Status/Level",
    "Gap",
    "Priority",
]


def _require_columns(dataframe, required_columns, filename):
    missing_columns = [column for column in required_columns if column not in dataframe]
    if missing_columns:
        raise ValueError(
            f"{filename} is missing required column(s): {', '.join(missing_columns)}"
        )


def load_inputs(
    agent_1_output_path=AGENT_1_OUTPUT_PATH,
    student_data_path=STUDENT_DATA_PATH,
    required_skills_path=REQUIRED_SKILLS_PATH,
):
    """Load Agent 1, explicit student data, and existing career requirements."""
    agent_1_path = Path(agent_1_output_path)
    student_data_file = Path(student_data_path)
    requirements_path = Path(required_skills_path)
    fallback_path = Path(FALLBACK_SKILLS_PATH)

    missing_files = [
        str(path)
        for path in (agent_1_path, student_data_file)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required input file(s) not found: " + ", ".join(missing_files)
        )

    if not requirements_path.is_file():
        if not fallback_path.is_file():
            raise FileNotFoundError(
                "Required skill source not found: "
                f"{requirements_path} or {fallback_path}"
            )
        requirements_path = fallback_path

    agent_1_profile = pd.read_csv(agent_1_path)
    student_data = pd.read_csv(student_data_file)
    required_skills = pd.read_csv(requirements_path)

    _require_columns(agent_1_profile, ["Student ID"], "student_profile_agent_output.csv")
    _require_columns(student_data, [STUDENT_ID_COLUMN], "student_dataset_cleaned.csv")
    if "skill" not in required_skills and "required_skill" in required_skills:
        required_skills = required_skills.rename(columns={"required_skill": "skill"})
    _require_columns(required_skills, ["skill"], requirements_path.name)
    return agent_1_profile, student_data, required_skills


def _normalise(value):
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _student_row(student_id, dataframe, filename, id_column=STUDENT_ID_COLUMN):
    if id_column not in dataframe:
        raise ValueError(f"{filename} is missing required column '{id_column}'.")
    rows = dataframe[dataframe[id_column].astype(str) == str(student_id)]
    if rows.empty:
        raise KeyError(f"Student ID '{student_id}' was not found in {filename}.")
    return rows.iloc[0]


def load_current_skills(student_id, agent_1_profile, student_data):
    """Return only explicitly named skill fields; never treat intelligence as skills."""
    current_skills = {}
    for dataframe, filename, id_column in (
        (agent_1_profile, "student_profile_agent_output.csv", "Student ID"),
        (student_data, "student_dataset_cleaned.csv", STUDENT_ID_COLUMN),
    ):
        row = _student_row(student_id, dataframe, filename, id_column)
        for column, value in row.items():
            if "skill" not in _normalise(column) or pd.isna(value):
                continue
            for skill in str(value).split(","):
                skill = skill.strip()
                if skill:
                    current_skills[_normalise(skill)] = skill
    return current_skills


def _required_skill_rows(required_skills, student_id):
    """Use the S1-specific downstream path as the career requirement source."""
    if "student_id" in required_skills:
        required_skills = required_skills[
            required_skills["student_id"].astype(str) == str(student_id)
        ]
    if required_skills.empty:
        raise KeyError(f"Student ID '{student_id}' has no required skills in the source.")
    return required_skills.drop_duplicates(subset=["skill"])


def assign_priority(gap, requirement_row):
    """Apply project priority logic to an actual gap and course availability."""
    if gap != "Gap":
        return "Low"
    if (str(requirement_row.get("status", "")) == "Skill Gap - No Course" or
            str(requirement_row.get("priority", "")) == "High"):
        return "High"
    return "Medium"


def build_skill_gap_table(student_id, agent_1_profile, student_data, required_skills):
    """Compare explicit current skills with the existing career-specific skill list."""
    current_skills = load_current_skills(student_id, agent_1_profile, student_data)
    rows = []
    for _, requirement in _required_skill_rows(required_skills, student_id).iterrows():
        required_skill = str(requirement["skill"])
        is_covered = _normalise(required_skill) in current_skills
        gap = "Covered" if is_covered else "Gap"
        rows.append({
            "Student ID": student_id,
            "Skill": required_skill,
            "Current Status/Level": current_skills.get(
                _normalise(required_skill), "Not Available"
            ),
            "Required Status/Level": "Required",
            "Gap": gap,
            "Priority": assign_priority(gap, requirement),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_skill_gap_table(skill_gap_table, output_path=OUTPUT_PATH):
    skill_gap_table.to_csv(output_path, index=False)
    return output_path


def print_summary(skill_gap_table):
    print("\nSKILL GAP AGENT SUMMARY")
    print("=" * 40)
    print("Total required skills:", len(skill_gap_table))
    print("Skills already covered:", (skill_gap_table["Gap"] == "Covered").sum())
    print("Skill gaps:", (skill_gap_table["Gap"] == "Gap").sum())
    for priority in ("High", "Medium", "Low"):
        count = ((skill_gap_table["Gap"] == "Gap") &
                 (skill_gap_table["Priority"] == priority)).sum()
        print(f"{priority}-priority gaps:", count)


def main():
    parser = argparse.ArgumentParser(description="Build the S1 skill-gap table.")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    try:
        inputs = load_inputs()
        skill_gap_table = build_skill_gap_table(STUDENT_ID, *inputs)
        print(skill_gap_table.to_string(index=False))
        print_summary(skill_gap_table)
        print("\nSkill-gap output saved to:", save_skill_gap_table(skill_gap_table, args.output))
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
