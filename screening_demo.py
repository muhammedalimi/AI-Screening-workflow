import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

ROLE_CONFIG = {
    "role_title": "Customer Support Specialist",
    "knockout_questions": [
        {
            "id": "work_auth",
            "question": "Are you authorized to work in the United States? (yes/no)",
            "required_answer": "yes"
        },
        {
            "id": "experience",
            "question": "Do you have at least 1 year of customer-facing experience? (yes/no)",
            "required_answer": "yes"
        }
    ],
    "assessment_questions": [
        "Tell me about a time you handled a difficult customer.",
        "How would you respond if a customer received a damaged order and was upset?",
        "Why are you interested in this customer support role?"
    ],
    "thresholds": {
        "advance": 80,
        "hold": 60
    }
}


def ask_yes_no(question: str) -> str:
    while True:
        answer = input(f"{question}\n> ").strip().lower()
        if answer in {"yes", "no"}:
            return answer
        print("Please answer yes or no.")


def ask_open_question(question: str) -> str:
    print(f"\n{question}")
    return input("> ").strip()


def run_knockout_screen(config: dict) -> tuple[bool, list[dict]]:
    results = []
    passed = True

    print("\n=== Eligibility Check ===")
    for item in config["knockout_questions"]:
        answer = ask_yes_no(item["question"])
        is_pass = answer == item["required_answer"]
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "passed": is_pass
        })
        if not is_pass:
            passed = False

    return passed, results


def collect_candidate_answers(config: dict) -> list[dict]:
    answers = []
    print("\n=== Candidate Assessment ===")
    for question in config["assessment_questions"]:
        response = ask_open_question(question)
        answers.append({
            "question": question,
            "answer": response
        })
    return answers


def build_scoring_prompt(candidate_name: str, role_title: str, answers: list[dict]) -> str:
    formatted_answers = "\n\n".join(
        [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in answers]
    )

    return f"""
You are an AI hiring assessment assistant.

Evaluate the candidate for the role: {role_title}.

Candidate name: {candidate_name}

Score the candidate on:
- communication
- empathy
- problem_solving
- professionalism
- role_fit

Use a 0-100 scale for each score.

Then calculate an overall_score from 0-100.

Return ONLY valid JSON in this exact structure:
{{
  "communication_score": 0,
  "empathy_score": 0,
  "problem_solving_score": 0,
  "professionalism_score": 0,
  "role_fit_score": 0,
  "overall_score": 0,
  "strengths": ["", ""],
  "concerns": ["", ""],
  "summary": ""
}}

Candidate responses:
{formatted_answers}
""".strip()


def score_candidate_with_gemini(candidate_name: str, role_title: str, answers: list[dict]) -> dict:
    prompt = build_scoring_prompt(candidate_name, role_title, answers)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=500,
            response_mime_type="application/json"
        )
    )

    text = response.text.strip()
    return json.loads(text)


def determine_recommendation(overall_score: int, thresholds: dict) -> str:
    if overall_score >= thresholds["advance"]:
        return "Advance"
    if overall_score >= thresholds["hold"]:
        return "Hold"
    return "Reject"


def save_result(result: dict, filename: str = "candidate_result.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    print("=== AI Screening Workflow Demo ===")
    print(f"Role: {ROLE_CONFIG['role_title']}")

    candidate_name = input("\nEnter candidate name:\n> ").strip()

    knockout_passed, knockout_results = run_knockout_screen(ROLE_CONFIG)

    if not knockout_passed:
        final_result = {
            "candidate_name": candidate_name,
            "role_title": ROLE_CONFIG["role_title"],
            "timestamp": datetime.now().isoformat(),
            "knockout_passed": False,
            "knockout_results": knockout_results,
            "recommendation": "Reject",
            "reason": "Candidate did not meet minimum eligibility requirements."
        }

        save_result(final_result)
        print("\n=== Final Decision ===")
        print("Recommendation: Reject")
        print("Reason: Candidate failed knockout criteria.")
        print("\nSaved result to candidate_result.json")
        return

    answers = collect_candidate_answers(ROLE_CONFIG)
    scoring = score_candidate_with_gemini(candidate_name, ROLE_CONFIG["role_title"], answers)
    recommendation = determine_recommendation(
        scoring["overall_score"],
        ROLE_CONFIG["thresholds"]
    )

    final_result = {
        "candidate_name": candidate_name,
        "role_title": ROLE_CONFIG["role_title"],
        "timestamp": datetime.now().isoformat(),
        "knockout_passed": True,
        "knockout_results": knockout_results,
        "answers": answers,
        "scores": scoring,
        "recommendation": recommendation
    }

    save_result(final_result)

    print("\n=== Final Decision ===")
    print(f"Recommendation: {recommendation}")
    print(f"Overall Score: {scoring['overall_score']}")
    print(f"Communication: {scoring['communication_score']}")
    print(f"Empathy: {scoring['empathy_score']}")
    print(f"Problem Solving: {scoring['problem_solving_score']}")
    print(f"Professionalism: {scoring['professionalism_score']}")
    print(f"Role Fit: {scoring['role_fit_score']}")

    print("\nStrengths:")
    for item in scoring["strengths"]:
        print(f"- {item}")

    print("\nConcerns:")
    for item in scoring["concerns"]:
        print(f"- {item}")

    print(f"\nSummary: {scoring['summary']}")
    print("\nSaved result to candidate_result.json")


if __name__ == "__main__":
    main()