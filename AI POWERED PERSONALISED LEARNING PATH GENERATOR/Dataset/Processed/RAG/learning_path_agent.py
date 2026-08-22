"""Agent 4: Build an ordered learning path from Agents 1-3 outputs."""

import argparse
import re
from pathlib import Path

import pandas as pd


PROFILE_PATH = "student_profile_agent_output.csv"
SKILL_GAP_PATH = "skill_gap_agent_output.csv"
RECOMMENDATION_PATH = "course_recommendation_agent_output.csv"
OUTPUT_PATH = "learning_path_agent_output.csv"

STAGE_ORDER = [
	"Programming",
	"Data",
	"Statistics",
	"Machine Learning",
	"Databases and Big Data",
	"Cloud and Tools",
	"Professional Skills",
]
STAGE_SKILLS = {
	"Programming": ["Python"],
	"Data": ["Data Integrity", "Data Preparation", "Data Mining", "Data Visualization"],
	"Statistics": ["Statistical Software", "Regressions", "Time Series Analysis"],
	"Machine Learning": ["Machine Learning", "Predictive Modeling", "Causal-Model Approaches", "Text Mining"],
	"Databases and Big Data": ["Sql", "Big Data", "Hadoop", "Apache Spark", "Hive", "Pig"],
	"Cloud and Tools": ["Cloud Platforms", "Aws", "Tableau"],
	"Professional Skills": ["Communication", "Stakeholder Engagement", "Strategic Thinking", "Problem Solving", "Project Management", "Teamwork", "Business Acumen", "Transparency", "Storytelling", "Iterative Development"],
}
OUTPUT_COLUMNS = [
	"Student ID", "Sequence", "Stage", "Skill", "Priority", "Course ID",
	"Course Name", "Institution", "Average Rating", "Review Count",
	"Course Score", "Course Status", "Learning Recommendation",
]


def _require_columns(dataframe, columns, filename):
	missing = [column for column in columns if column not in dataframe.columns]
	if missing:
		raise ValueError(f"{filename} is missing required column(s): {', '.join(missing)}")


def load_inputs(profile_path=PROFILE_PATH, skill_gap_path=SKILL_GAP_PATH,
				recommendation_path=RECOMMENDATION_PATH):
	paths = {
		"student_profile_agent_output.csv": Path(profile_path),
		"skill_gap_agent_output.csv": Path(skill_gap_path),
		"course_recommendation_agent_output.csv": Path(recommendation_path),
	}
	missing = [str(path) for path in paths.values() if not path.is_file()]
	if missing:
		raise FileNotFoundError("Required input file(s) not found: " + ", ".join(missing))

	profile = pd.read_csv(paths["student_profile_agent_output.csv"])
	skill_gaps = pd.read_csv(paths["skill_gap_agent_output.csv"])
	recommendations = pd.read_csv(paths["course_recommendation_agent_output.csv"])
	_require_columns(profile, ["Student ID", "Dominant Intelligence"], "student_profile_agent_output.csv")
	_require_columns(skill_gaps, ["Student ID", "Skill", "Gap", "Priority"], "skill_gap_agent_output.csv")
	_require_columns(recommendations, ["Student ID", "Skill", "Priority", "Course Status"], "course_recommendation_agent_output.csv")
	return profile, skill_gaps, recommendations


def _normalise(value):
	return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _learning_recommendation(profile):
	dominant = str(profile["Dominant Intelligence"])
	if _normalise(dominant) == "naturalist":
		return "Use examples, categorization, comparisons and pattern recognition."
	return f"Use learning activities suited to {dominant} intelligence."


def _stage_lookup():
	return {
		_normalise(skill): (stage, index)
		for stage in STAGE_ORDER
		for index, skill in enumerate(STAGE_SKILLS[stage])
	}


def build_learning_path(profile, skill_gaps, recommendations):
	profile_ids = profile["Student ID"].dropna().astype(str).unique()
	if len(profile_ids) != 1:
		raise ValueError("student_profile_agent_output.csv must contain one Student ID.")
	student_id = profile_ids[0]
	student_profile = profile.iloc[0]
	student_gaps = skill_gaps[
		(skill_gaps["Student ID"].astype(str) == student_id) &
		(skill_gaps["Gap"].astype(str).str.casefold() == "gap")
	].copy()
	if student_gaps.empty:
		raise KeyError(f"No skill gaps found for Student ID '{student_id}'.")

	recommendations = recommendations[
		recommendations["Student ID"].astype(str) == student_id
	]
	recommendation_by_skill = {
		_normalise(row["Skill"]): row
		for _, row in recommendations.iterrows()
	}
	stage_lookup = _stage_lookup()
	student_gaps["_stage"] = student_gaps["Skill"].map(
		lambda skill: stage_lookup.get(_normalise(skill), ("Unmapped", len(STAGE_ORDER))) [0]
	)
	student_gaps["_stage_order"] = student_gaps["Skill"].map(
		lambda skill: stage_lookup.get(_normalise(skill), ("Unmapped", len(STAGE_ORDER))) [1]
	)
	student_gaps = student_gaps.sort_values(["_stage_order"])

	recommendation_text = _learning_recommendation(student_profile)
	rows = []
	for sequence, (_, gap) in enumerate(student_gaps.iterrows(), start=1):
		course = recommendation_by_skill.get(_normalise(gap["Skill"]))
		row = {
			"Student ID": student_id,
			"Sequence": sequence,
			"Stage": stage_lookup.get(_normalise(gap["Skill"]), ("Unmapped", 0))[0],
			"Skill": gap["Skill"],
			"Priority": gap["Priority"],
			"Course ID": "",
			"Course Name": "",
			"Institution": "",
			"Average Rating": "",
			"Review Count": "",
			"Course Score": "",
			"Course Status": "Skill Gap - No Course",
			"Learning Recommendation": recommendation_text,
		}
		if course is not None and str(course["Course Status"]) != "No Course Available":
			for output_column, source_column in {
				"Course ID": "Course ID", "Course Name": "Course Name",
				"Institution": "Institution", "Average Rating": "Average Rating",
				"Review Count": "Review Count", "Course Score": "Course Score",
			}.items():
				row[output_column] = course[source_column]
			row["Course Status"] = "Course Available"
		rows.append(row)
	return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_learning_path(learning_path, output_path=OUTPUT_PATH):
	learning_path.to_csv(output_path, index=False)
	return output_path


def print_summary(learning_path):
	available = learning_path["Course Status"] == "Course Available"
	high = learning_path["Priority"].astype(str).str.casefold() == "high"
	medium = learning_path["Priority"].astype(str).str.casefold() == "medium"
	print("PERSONALIZED LEARNING PATH AGENT")
	print("================================")
	print("Student ID:", learning_path["Student ID"].iloc[0])
	print("Total learning steps:", len(learning_path))
	print("Courses available:", available.sum())
	print("Skill gaps:", (~available).sum())
	print("High-priority steps:", high.sum())
	print("Medium-priority steps:", medium.sum())
	print("\nOrdered learning path")
	print(learning_path.to_string(index=False))


def main():
	parser = argparse.ArgumentParser(description="Build the S1 personalized learning path.")
	parser.add_argument("--output", default=OUTPUT_PATH)
	args = parser.parse_args()
	try:
		learning_path = build_learning_path(*load_inputs())
		print_summary(learning_path)
		print("\nLearning path saved to:", save_learning_path(learning_path, args.output))
	except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()
