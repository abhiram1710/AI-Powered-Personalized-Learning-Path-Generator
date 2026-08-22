"""Agent 5: Build deterministic quiz and practice recommendations."""

import argparse
from pathlib import Path

import pandas as pd


INPUT_PATH = "learning_path_agent_output.csv"
OUTPUT_PATH = "quiz_agent_output.csv"
REQUIRED_COLUMNS = [
	"Student ID", "Sequence", "Stage", "Skill", "Priority",
	"Course Name", "Course Status",
]
QUIZ_COLUMNS = ["Quiz Type", "Number of Questions", "Practice Level", "Quiz Status"]


def load_inputs(input_path=INPUT_PATH):
	"""Load and validate the ordered learning path."""
	path = Path(input_path)
	if not path.is_file():
		raise FileNotFoundError(f"Input file not found: {path}")
	learning_path = pd.read_csv(path)
	missing = [column for column in REQUIRED_COLUMNS if column not in learning_path.columns]
	if missing:
		raise ValueError(
			"learning_path_agent_output.csv is missing required column(s): "
			+ ", ".join(missing)
		)
	return learning_path


def _quiz_type(stage):
	quiz_types = {
		"Programming": "Code writing and debugging",
		"Data": "Data analysis and application",
		"Statistics": "Calculation and concept quiz",
		"Machine Learning": "Modeling concepts and application",
		"Databases and Big Data": "Query and systems application",
		"Cloud and Tools": "Tools and workflow application",
		"Professional Skills": "Scenario-based assessment",
	}
	return quiz_types.get(stage, "Skill knowledge and application")


def _practice_details(priority, course_status):
	priority_key = str(priority).strip().casefold()
	if priority_key == "high":
		return "Advanced practice", 10
	if priority_key == "medium":
		return "Standard practice", 7
	return "Foundational practice", 5


def build_quiz_plan(learning_path):
	"""Create one quiz or practice recommendation for every learning step."""
	quiz_plan = learning_path.copy()
	quiz_types = []
	question_counts = []
	practice_levels = []
	statuses = []
	for _, step in quiz_plan.iterrows():
		practice_level, question_count = _practice_details(
			step["Priority"], step["Course Status"]
		)
		quiz_types.append(_quiz_type(step["Stage"]))
		question_counts.append(question_count)
		practice_levels.append(practice_level)
		if str(step["Course Status"]).strip() == "Course Available":
			statuses.append("Quiz Recommended")
		else:
			statuses.append("Practice Required - No Course")

	quiz_plan["Quiz Type"] = quiz_types
	quiz_plan["Number of Questions"] = question_counts
	quiz_plan["Practice Level"] = practice_levels
	quiz_plan["Quiz Status"] = statuses
	return quiz_plan


def save_quiz_plan(quiz_plan, output_path=OUTPUT_PATH):
	"""Save the generated quiz and practice plan."""
	quiz_plan.to_csv(output_path, index=False)
	return output_path


def print_summary(quiz_plan):
	"""Print counts for quiz and no-course practice steps."""
	recommended = quiz_plan["Quiz Status"] == "Quiz Recommended"
	practice_required = quiz_plan["Quiz Status"] == "Practice Required - No Course"
	high = quiz_plan["Priority"].astype(str).str.casefold() == "high"
	medium = quiz_plan["Priority"].astype(str).str.casefold() == "medium"
	print("QUIZ AGENT")
	print("=========")
	print("Total learning steps:", len(quiz_plan))
	print("Quiz recommendations:", recommended.sum())
	print("Practice-required steps:", practice_required.sum())
	print("High-priority quiz steps:", (recommended & high).sum())
	print("Medium-priority quiz steps:", (recommended & medium).sum())


def main():
	parser = argparse.ArgumentParser(description="Build quizzes and practice for the learning path.")
	parser.add_argument("--input", default=INPUT_PATH)
	parser.add_argument("--output", default=OUTPUT_PATH)
	args = parser.parse_args()
	try:
		quiz_plan = build_quiz_plan(load_inputs(args.input))
		print_summary(quiz_plan)
		print("\nGenerated quiz/practice table")
		print(quiz_plan.to_string(index=False))
		print("\nQuiz plan saved to:", save_quiz_plan(quiz_plan, args.output))
	except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()
