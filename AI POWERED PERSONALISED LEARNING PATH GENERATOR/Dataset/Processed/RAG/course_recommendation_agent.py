"""Agent 3: Recommend the highest-ranked course for each skill gap."""

import argparse
import re
from pathlib import Path

import pandas as pd


STUDENT_PROFILE_PATH = "student_profile_agent_output.csv"
SKILL_GAP_PATH = "skill_gap_agent_output.csv"
RANKED_COURSES_PATH = "ranked_courses.csv"
OUTPUT_PATH = "course_recommendation_agent_output.csv"

REQUIRED_PROFILE_COLUMNS = ["Student ID"]
REQUIRED_GAP_COLUMNS = ["Student ID", "Skill", "Gap", "Priority"]
REQUIRED_COURSE_COLUMNS = [
	"required_skill",
	"course_id",
	"course_name",
	"institution",
	"average_rating",
	"review_count",
	"course_score",
]
OUTPUT_COLUMNS = [
	"Student ID",
	"Skill",
	"Priority",
	"Course ID",
	"Course Name",
	"Institution",
	"Average Rating",
	"Review Count",
	"Course Score",
	"Course Status",
]


def _require_columns(dataframe, columns, filename):
	missing = [column for column in columns if column not in dataframe.columns]
	if missing:
		raise ValueError(
			f"{filename} is missing required column(s): {', '.join(missing)}"
		)


def load_inputs(
	student_profile_path=STUDENT_PROFILE_PATH,
	skill_gap_path=SKILL_GAP_PATH,
	ranked_courses_path=RANKED_COURSES_PATH,
):
	"""Load Agent 1, Agent 2, and ranked-course outputs."""
	paths = {
		"student_profile_agent_output.csv": Path(student_profile_path),
		"skill_gap_agent_output.csv": Path(skill_gap_path),
		"ranked_courses.csv": Path(ranked_courses_path),
	}
	missing = [str(path) for path in paths.values() if not path.is_file()]
	if missing:
		raise FileNotFoundError("Required input file(s) not found: " + ", ".join(missing))

	student_profile = pd.read_csv(paths["student_profile_agent_output.csv"])
	skill_gaps = pd.read_csv(paths["skill_gap_agent_output.csv"])
	ranked_courses = pd.read_csv(paths["ranked_courses.csv"])
	_require_columns(student_profile, REQUIRED_PROFILE_COLUMNS, "student_profile_agent_output.csv")
	_require_columns(skill_gaps, REQUIRED_GAP_COLUMNS, "skill_gap_agent_output.csv")
	_require_columns(ranked_courses, REQUIRED_COURSE_COLUMNS, "ranked_courses.csv")
	return student_profile, skill_gaps, ranked_courses


def normalise_skill(skill):
	"""Normalize case, whitespace, hyphens, and punctuation for skill matching."""
	return re.sub(r"[^a-z0-9]", "", str(skill).lower())


def _best_course(skill, ranked_courses):
	matching = ranked_courses[
		ranked_courses["required_skill"].map(normalise_skill) == normalise_skill(skill)
	].copy()
	if matching.empty:
		return None
	matching["course_score"] = pd.to_numeric(matching["course_score"], errors="coerce")
	matching = matching.dropna(subset=["course_score"])
	if matching.empty:
		return None
	return matching.sort_values("course_score", ascending=False).iloc[0]


def build_recommendations(student_profile, skill_gaps, ranked_courses):
	"""Create one recommendation row for every actual skill gap."""
	student_ids = student_profile["Student ID"].dropna().astype(str).unique()
	if len(student_ids) != 1:
		raise ValueError("student_profile_agent_output.csv must contain one Student ID.")
	student_id = student_ids[0]
	skill_gaps = skill_gaps[
		skill_gaps["Student ID"].astype(str) == student_id
		].copy()
	if skill_gaps.empty:
		raise KeyError(f"No skill-gap data found for Student ID '{student_id}'.")
	skill_gaps = skill_gaps[
		skill_gaps["Gap"].astype(str).str.strip().str.casefold() == "gap"
	]

	priority_order = {"high": 0, "medium": 1, "low": 2}
	skill_gaps["_priority_order"] = (
		skill_gaps["Priority"].astype(str).str.lower().map(priority_order).fillna(3)
	)
	skill_gaps = skill_gaps.sort_values("_priority_order")
	rows = []
	for _, gap in skill_gaps.iterrows():
		skill = gap["Skill"]
		course = _best_course(skill, ranked_courses)
		row = {
			"Student ID": student_id,
			"Skill": skill,
			"Priority": gap["Priority"],
			"Course ID": "",
			"Course Name": "",
			"Institution": "",
			"Average Rating": "",
			"Review Count": "",
			"Course Score": "",
			"Course Status": "No Course Available",
		}
		if course is not None:
			row.update({
				"Course ID": course["course_id"],
				"Course Name": course["course_name"],
				"Institution": course["institution"],
				"Average Rating": course["average_rating"],
				"Review Count": course["review_count"],
				"Course Score": course["course_score"],
				"Course Status": "Course Recommended",
			})
		rows.append(row)
	return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_recommendations(recommendations, output_path=OUTPUT_PATH):
	recommendations.to_csv(output_path, index=False)
	return output_path


def print_summary(recommendations):
	"""Print counts for course-backed and unavailable skill gaps."""
	with_courses = recommendations["Course Status"] == "Course Recommended"
	high = recommendations["Priority"].astype(str).str.casefold() == "high"
	medium = recommendations["Priority"].astype(str).str.casefold() == "medium"
	print("\nCOURSE RECOMMENDATION AGENT SUMMARY")
	print("=" * 45)
	print("Total skill gaps:", len(recommendations))
	print("Skill gaps with course recommendations:", with_courses.sum())
	print("Skill gaps without courses:", (~with_courses).sum())
	print("High-priority recommendations:", (with_courses & high).sum())
	print("Medium-priority recommendations:", (with_courses & medium).sum())


def main():
	parser = argparse.ArgumentParser(description="Recommend courses for S1 skill gaps.")
	parser.add_argument("--output", default=OUTPUT_PATH)
	args = parser.parse_args()
	try:
		inputs = load_inputs()
		recommendations = build_recommendations(*inputs)
		print(recommendations.to_string(index=False))
		print_summary(recommendations)
		print("\nCourse recommendations saved to:", save_recommendations(recommendations, args.output))
	except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()
