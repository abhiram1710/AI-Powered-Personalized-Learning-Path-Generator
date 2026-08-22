"""Agent 1: Build a structured profile for one student."""

import argparse
from pathlib import Path

import pandas as pd


STUDENT_PROFILES_PATH = r"..\student\student_profiles.csv"
STUDENT_INTELLIGENCE_PATH = r"..\student\student_intelligence_profiles.csv"
OUTPUT_PATH = "student_profile_agent_output.csv"
STUDENT_ID_COLUMN = "Student_ID"
INTELLIGENCE_DIMENSIONS = {
	"linguistic",
	"musical",
	"bodily",
	"logicalmathematical",
	"spatialvisualization",
	"interpersonal",
	"intrapersonal",
	"naturalist",
}


def load_student_data(
	student_profiles_path=STUDENT_PROFILES_PATH,
	intelligence_profiles_path=STUDENT_INTELLIGENCE_PATH,
):
	"""Load the two source datasets used by the student profile agent."""
	profiles_path = Path(student_profiles_path)
	intelligence_path = Path(intelligence_profiles_path)

	missing_files = [
		str(path)
		for path in (profiles_path, intelligence_path)
		if not path.is_file()
	]
	if missing_files:
		raise FileNotFoundError(
			"Required input file(s) not found: " + ", ".join(missing_files)
		)

	return pd.read_csv(profiles_path), pd.read_csv(intelligence_path)


def _validate_student_id(dataframe, dataframe_name):
	if STUDENT_ID_COLUMN not in dataframe.columns:
		raise ValueError(
			f"{dataframe_name} is missing the required '{STUDENT_ID_COLUMN}' column."
		)


def _clean_value(value):
	return value.item() if hasattr(value, "item") else value


def _normalise_column_name(column):
	return "".join(character.lower() for character in str(column) if character.isalnum())


def _find_dominant_column(columns):
	for column in columns:
		normalised_column = _normalise_column_name(column)
		if "dominant" in normalised_column and "intelligence" in normalised_column:
			return column
	return None


def _find_intelligence_score_columns(columns):
	"""Return only columns representing an intelligence dimension score."""
	score_columns = []
	for column in columns:
		normalised_column = _normalise_column_name(column)
		base_name = normalised_column.removesuffix("score")
		if base_name in INTELLIGENCE_DIMENSIONS:
			score_columns.append(column)
	return score_columns


def combine_student_profile(student_id, student_profiles, intelligence_profiles):
	"""Combine all available source fields and derive intelligence summary fields."""
	_validate_student_id(student_profiles, "student_profiles.csv")
	_validate_student_id(intelligence_profiles, "student_intelligence_profiles.csv")

	profile_rows = student_profiles[
		student_profiles[STUDENT_ID_COLUMN].astype(str) == str(student_id)
	]
	intelligence_rows = intelligence_profiles[
		intelligence_profiles[STUDENT_ID_COLUMN].astype(str) == str(student_id)
	]

	if profile_rows.empty and intelligence_rows.empty:
		raise KeyError(f"Student ID '{student_id}' was not found in either input file.")

	profile = {"Student ID": str(student_id)}
	if not profile_rows.empty:
		for column, value in profile_rows.iloc[0].items():
			if column != STUDENT_ID_COLUMN:
				profile[column.replace("_", " ")] = _clean_value(value)

	if not intelligence_rows.empty:
		intelligence_row = intelligence_rows.iloc[0]
		for column, value in intelligence_row.items():
			if column == STUDENT_ID_COLUMN:
				continue
			clean_column = column.replace("_", " ")
			profile[clean_column] = _clean_value(value)

		dominant_column = _find_dominant_column(intelligence_row.index)
		if dominant_column is not None:
			profile["Dominant Intelligence"] = _clean_value(
				intelligence_row[dominant_column]
			)

		score_columns = _find_intelligence_score_columns(intelligence_row.index)
		intelligence_scores = {
			column: _clean_value(intelligence_row[column])
			for column in score_columns
			if pd.api.types.is_number(intelligence_row[column])
		}
		if dominant_column is None and intelligence_scores:
			dominant_column = max(intelligence_scores, key=intelligence_scores.get)
			profile["Dominant Intelligence"] = dominant_column.replace("_", " ")

		if "Dominant Intelligence" in profile:
			dominant_name = _normalise_column_name(profile["Dominant Intelligence"])
			matching_score = next(
				(
					column
					for column in score_columns
					if _normalise_column_name(column).removesuffix("score")
					== dominant_name
				),
				None,
			)
			if matching_score is not None and matching_score in intelligence_scores:
				profile["Intelligence Score"] = intelligence_scores[matching_score]

	return profile


def build_student_profile(student_id, student_profiles_path=STUDENT_PROFILES_PATH,
						  intelligence_profiles_path=STUDENT_INTELLIGENCE_PATH):
	"""Load source data and return one clean profile as a one-row DataFrame."""
	student_profiles, intelligence_profiles = load_student_data(
		student_profiles_path, intelligence_profiles_path
	)
	profile = combine_student_profile(
		student_id, student_profiles, intelligence_profiles
	)
	return pd.DataFrame([profile])


def print_student_profile(profile):
	"""Print a profile in a readable key-value format."""
	print("\nStudent Profile")
	print("=" * 40)
	for field, value in profile.iloc[0].items():
		print(f"{field}: {value}")


def save_student_profile(profile, output_path=OUTPUT_PATH):
	"""Save the structured profile to a CSV file."""
	profile.to_csv(output_path, index=False)
	return output_path


def main():
	parser = argparse.ArgumentParser(description="Build a student learning profile.")
	parser.add_argument("student_id", help="Student ID, for example S1")
	parser.add_argument(
		"--output", default=OUTPUT_PATH, help="Output CSV path (default: %(default)s)"
	)
	args = parser.parse_args()

	try:
		profile = build_student_profile(args.student_id)
		print_student_profile(profile)
		output_path = save_student_profile(profile, args.output)
		print(f"\nProfile saved to: {output_path}")
	except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()
