# Project Summary

## Overview
This project turns a student profile and career target into an ordered learning path with courses, practice guidance, and knowledge-base support.

## Architecture flow
Student input -> Profile -> Skill gap -> Course recommendation -> Learning path -> Quiz/practice -> Final personalization -> RAG assistant -> Dashboard.

## Agents
1. **Student Profile Agent:** produces a clean profile and intelligence summary.
2. **Skill Gap Agent:** identifies required skills and compares them with explicitly available technical skills.
3. **Course Recommendation Agent:** recommends the best ranked course per gap, or records no course.
4. **Learning Path Agent:** orders skills across Programming, Data, Statistics, Machine Learning, Databases and Big Data, Cloud and Tools, and Professional Skills.
5. **Quiz Agent:** assigns category-based quiz types, question counts, and practice levels.
6. **Final Personalization Agent:** combines the ordered learning path with quiz/practice fields.

## RAG
The FAISS index searches 41 knowledge chunks using Sentence Transformers. The Streamlit assistant retrieves relevant project knowledge for questions about the path.

## Input -> Processing -> Output
Student and career CSVs -> six deterministic pandas agents -> agent output CSVs -> dashboard and final personalized path.

## Key features
- Student ID selection
- Intelligence profile visualization
- Skill-gap table
- Course recommendations and unavailable-course list
- Ordered 31-step learning path
- Quiz and practice table
- RAG learning assistant
- CSV download

## Example S1 workflow
S1 is an Astronomer with Naturalist as the dominant intelligence. The dashboard presents the generated skill gaps, course-backed steps, five unavailable-course gaps, ordered stages, and pattern-based learning guidance.

## Technologies
Python 3.13, pandas, Streamlit, FAISS, Sentence Transformers, NumPy, and CSV files.

## Future scope
Capture real skill assessments, track completion, score quizzes, refresh course data, and personalize recommendations from learner feedback.
