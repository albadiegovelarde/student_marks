import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import yaml
import re
import os
import gc

from retriever import StudentRetriever


LLM_MODEL_NAME = "google/flan-t5-large"
MAX_TOKENS = 200

SKILLS_YAML = "student_marks/skills.yaml"

## LLM 
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
text_gen = pipeline(
    "text2text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_TOKENS,
    device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1
)

# Skill descriptions
with open(SKILLS_YAML, "r", encoding="utf-8") as f:
    skills_data = yaml.safe_load(f)

skill_descriptions = {k: v["description"] for k, v in skills_data.items()}

# Retriever
retriever = StudentRetriever()


def parse_user_input(user_input: str):
    """
    Extracts the student ID and skill from a user question.

    Example:
        Input: "Quiero que me hagas un resumen del alumno student_1 de la skill math"
        Output: ("student_1", "math")

    Args:
        user_input (str): The user-provided question or query.

    Returns:
        tuple:
            student_id (str or None): Extracted student identifier or None if not found.
            skill (str or None): Extracted skill name or None if not found.
    """
    student_match = re.search(r"student[_\s]?(\d+)", user_input, re.IGNORECASE)
    skill_match = re.search(r"skill\s+(\w+)", user_input, re.IGNORECASE)

    student_id = f"student_{student_match.group(1)}" if student_match else None
    skill = skill_match.group(1) if skill_match else None

    return student_id, skill

def generate_summary_prompt(chunks, skill_description, user_input):
    """
    Generates a prompt for the language model using relevant text chunks and the skill description.

    Args:
        chunks (list of dict): A list of chunks with relevant student info.
        skill_description (str): Description of the skill to summarize.
        user_input: User query.

    Returns:
        str: A formatted prompt string suitable for input to the LLM.
    """
    text_context = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""
    Skill description: {skill_description}

    Relevant information about the student:
    {text_context}

    {user_input}
    """
    return prompt

def generate_student_summary(user_input: str):
    """
    Processes a user query to generate a concise summary of a student's performance for a specific skill.

    Steps:
        1. Parse the user input to extract student_id and skill.
        2. Validate the extracted skill exists in the YAML.
        3. Retrieve the top-k relevant chunks for the student and skill.
        4. Generate a prompt combining the skill description and relevant chunks.
        5. Call the language model to produce a summary.
    
    Args:
        user_input (str): The user's question or request.

    Returns:
        str: Generated summary text or an error message if extraction or retrieval fails.
    """
    student_id, skill = parse_user_input(user_input)

    if not student_id or not skill:
        return "I could not identify student_id or skill in your question."

    skill_description = skill_descriptions.get(skill)
    if not skill_description:
        return f"Skill '{skill}' not found in YAML."

    # Retrieval
    chunks = retriever.retrieve(skill_description, student_id=student_id, top_k=3)
    if not chunks:
        return f"No chunks for {student_id} and {skill}."

    # Generate prompt
    prompt = generate_summary_prompt(chunks, skill_description, user_input)

    # Call LLM
    summary = text_gen(prompt, max_new_tokens=MAX_TOKENS)[0]["generated_text"]
    return summary.strip()


### GRADIO
iface = gr.Interface(
    fn=generate_student_summary,
    inputs=gr.Textbox(lines=2, label="Question"),
    outputs=gr.Textbox(label="Answer", lines=3)
)

if __name__ == "__main__":
    iface.launch()