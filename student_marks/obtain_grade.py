import yaml
from typing import List, Dict
from retriever import StudentRetriever
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import pandas as pd


STUDENTS_IDS = ["student_1", "student_2", "student_3", "student_4"]  # student ids
SKILLS_YAML = "./skills.yaml"

LLM_MODEL_NAME = "google/flan-t5-large"
MAX_TOKENS = 200

TOP_K = 3  # Number of chunks to retrive


tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
text_gen = pipeline(
    "text2text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_TOKENS,
    device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1
)

def load_skills(yaml_path: str) -> List[Dict]:
    """
    Load skill evaluation criteria from a YAML file.

    Args:
        yaml_path (str): Path to the YAML file containing skill definitions.

    Returns:
        List[Dict]: A list of dictionaries, each representing a skill with its description, weight, and prompt instructions.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def generate_prompt(instructions: str, chunks: List[Dict]) -> str:
    """
    Build a prompt for the LLM to generate a grade based on the skill instructions
    and the relevant student chunks.

    Args:
        instructions (str): Instructions describing how to evaluate the skill.
        chunks (List[Dict]): List of dictionaries containing relevant text chunks for the student.

    Returns:
        str: A formatted prompt string ready to be passed to the LLM.
    """
    text_context = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""
            Instructions: {instructions}

            Relevant information about the student:
            {text_context}

            Return an integer number between 1 and 10 representing the grade.
            """
    return prompt

def generate_explanation_prompt(instructions: str, chunks: List[Dict], grade: int, skill_name: str) -> str:
    """
    Build a prompt for the LLM to explain why a certain grade was given.

    Args:
        instructions (str): Instructions describing how the skill was evaluated.
        chunks (List[Dict]): List of dictionaries containing the relevant text chunks for the student.
        grade (int): The grade previously assigned to the student for this skill.
        skill_name (str): Name of the skill being evaluated.

    Returns:
        str: A formatted prompt asking the LLM to provide a clear explanation in plain text.
    """
    text_context = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""
    The student received the grade {grade} for the competence "{skill_name}".

    Instructions used for evaluation:
    {instructions}

    Chunks considered:
    {text_context}

    Provide a clear and detailed explanation (3-5 sentences) of why this grade was given,
    linking the evaluation instructions with the retrieved information.
    Do NOT return JSON, only plain text.
    """
    return prompt


def call_llm(prompt: str) -> str:
    """
    Generate text output from the LLM given a prompt.

    Args:
        prompt (str): The input prompt to send to the LLM.

    Returns:
        Union[int, str]: The LLM output.
    """
    raw_output = text_gen(prompt, max_new_tokens=MAX_TOKENS)[0]["generated_text"]
    try:
        return int(raw_output)
    except:
        return raw_output

def compute_global_mark(grades: Dict[str, Dict], skills: Dict[str, Dict]) -> float:
    """
    Compute the global grade for a student based on individual skill grades.

    Args:
        grades (Dict[str, Dict]): Dictionary of grades per skill for a student. 
                                  Each value should include 'grade' and other info.
        skills (Dict[str, Dict]): Dictionary of skill metadata, including 'weight' for each skill.

    Returns:
        float: The weighted average grade rounded to 2 decimals, or None if no valid grades are available.
    """
    total_weight = 0
    weighted_sum = 0

    for skill_name, result in grades.items():
        # saltar la clave global_mark
        if skill_name == "global grade":
            continue  

        grade = result["grade"]
        weight = skills[skill_name]["weight"]
        
        if grade is not None:
            weighted_sum += grade * weight
            total_weight += weight

    if total_weight == 0:
        return None
    return round(weighted_sum / total_weight, 2)


if __name__ == "__main__":
    skills = load_skills(SKILLS_YAML)
    retriever = StudentRetriever()

    student_grades = {}

    for student_id in STUDENTS_IDS:
        student_grades[student_id] = {}

        for skill_name, skill_data in skills.items():
            description = skill_data.get("description", "")
            instructions = skill_data.get("prompt_instructions", "")
            weight = skill_data.get("weight", "Error")

            chunks = retriever.retrieve(description, student_id=student_id, top_k=TOP_K)
            if not chunks:
                print(f"No chunks found for student {student_id} and skill '{description}'")
                continue

            grade_prompt = generate_prompt(instructions, chunks)
            grade_output = call_llm(grade_prompt)
            print(grade_output)

            explanation_prompt = generate_explanation_prompt(instructions, chunks, grade_output, skill_name)
            explanation_output = call_llm(explanation_prompt)
            print(explanation_output)

            student_grades[student_id][skill_name] = {
                "grade": grade_output,
                "explanation": explanation_output.strip(),
                "relevant_chunks": chunks,
                "weight": weight
            }

        global_mark = compute_global_mark(student_grades[student_id], skills)
        student_grades[student_id]["global grade"] = global_mark

    rows = []
    for student_id, skills in student_grades.items():
        global_grade = skills.get("global grade")
        for skill, result in skills.items():
            if skill == "global grade":
                continue
            rows.append({
                "student_id": student_id,
                "skill": skill,
                "grade": result.get("grade"),
                "explanation": result.get("explanation"),
                "relevant_chunks": [chunk['text'] for chunk in result.get("relevant_chunks")],
                "global_grade": global_grade
                })
    student_grades_df = pd.DataFrame(rows)