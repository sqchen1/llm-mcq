import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import os
import logging
from concurrent.futures import ThreadPoolExecutor

# =============================
# Logging configuration
# =============================
logging.basicConfig(
    filename='model_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================
# Model setup
# =============================
API_KEY = "place_holder"
API_BASE = "https://api.openai.com/v1"

gpt_llm_1 = ChatOpenAI(
    openai_api_base=API_BASE,
    openai_api_key=API_KEY,
    model_name='gpt-4.1',
    temperature=0.2,
    top_p=0.8,
)

gpt_llm_2 = ChatOpenAI(
    openai_api_base=API_BASE,
    openai_api_key=API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.2,
    top_p=0.8,
)

gpt_llm_3 = ChatOpenAI(
    openai_api_base=API_BASE,
    openai_api_key=API_KEY,
    model_name='gpt-4o-mini',
    temperature=0.2,
    top_p=0.8,
)

# =============================
# Model list
# =============================
llm_configs = [gpt_llm_1, gpt_llm_2, gpt_llm_3]


# =============================
# Prompt builder for MCQs
# =============================
def build_choice_prompt(question: str, format_req: str) -> str:
    """Build prompt for multiple-choice questions only"""
    return f"""
Please choose the correct answer (A/B/C/D) based on the following prompt.

[Question]
{question}

[Requirements]
1. Output ONLY the letter of your choice ({format_req})
2. Do not provide explanations or reasoning
3. The final answer must be one capital letter: A / B / C / D
"""


# =============================
# Process model responses
# =============================
def process_answer(response) -> str:
    """Process and validate model responses"""
    try:
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        answer = answer.upper()
        if len(answer) == 1 and answer in ("A", "B", "C", "D"):
            return answer
        else:
            error_msg = f"⚠️ Format error: {answer} (expected A/B/C/D)"
            logging.warning(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"❌ Error processing response: {str(e)}"
        logging.error(error_msg)
        return error_msg


# =============================
# Parallel model invocation
# =============================
def parallel_call_models(prompt: str, models: list) -> dict:
    """Call all models in parallel and collect responses"""
    results = {}

    def _call_single_model(model):
        try:
            response = model.invoke(prompt)
            return model.model_name, response
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            logging.error(f"[{model.model_name}] {error_msg}")
            return model.model_name, error_msg

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [executor.submit(_call_single_model, model) for model in models]
        for future in futures:
            model_name, result = future.result()
            results[model_name] = result

    return results


# =============================
# Main data processing
# =============================
def process_data():
    input_file = "/Users/kevinchen/Desktop/test_qa.xlsx"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

    df = pd.read_excel(input_file)

    # Create result columns
    for model in llm_configs:
        df[f"{model.model_name}_Answer"] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing MCQs"):
        question = row["Question"]
        format_req = row["FormatRequirement"]

        prompt = build_choice_prompt(question, format_req)

        answers = parallel_call_models(prompt, llm_configs)

        for model_name, answer_or_error in answers.items():
            if isinstance(answer_or_error, str) and answer_or_error.startswith(
                ("API call failed", "Error processing", "Format error")
            ):
                final_answer = answer_or_error
            else:
                final_answer = process_answer(answer_or_error)
            df.at[idx, f"{model_name}_Answer"] = final_answer

    df.to_excel("output_with_answers.xlsx", index=False)
    print("✅ MCQ results saved to output_with_answers.xlsx")


# =============================
# Entry point
# =============================
if __name__ == "__main__":
    process_data()
