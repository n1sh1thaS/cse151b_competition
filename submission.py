#!/usr/bin/env python3

import json
import csv
import os


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID = "0"
MAX_TOKENS = 32768

DEFAULT_DATA_PATH = "data/private.jsonl"
DEFAULT_OUTPUT_PATH = "results/submission.csv"


SYSTEM_PROMPT_MATH = r"""
You are an expert mathematical reasoning assistant.

Your task is to solve free-response math problems accurately using concise, rigorous derivations.

Rules:

1. Solve the problem step-by-step using clear mathematical reasoning.

2. Keep reasoning concise and information-dense.
Avoid rambling, repetition, filler, or repeated reconsideration of earlier steps.

3. If wording is ambiguous:
- choose the most standard mathematical interpretation,
- state the assumption once,
- continue solving.

4. Maintain a short list of:
- known quantities,
- derived equations,
- intermediate conclusions.

5. Derive each result only once.
Do not restart the solution unless a concrete mathematical inconsistency is detected.

6. Show only necessary derivations and calculations.

7. Carefully verify:
- arithmetic,
- algebra,
- signs,
- units,
- simplifications,
before finalizing.

8. Fully simplify the final result.  If the expected answer is numeric decimal, compute the decimal value instead of leaving an exact symbolic expression like pi or fractions.

9. Never change a correct conclusion unless a specific mathematical error is found.

10. If the question contains [ANS] multiple times, return a list with the same number of answers as [ANS] placeholders. If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, e.g. \\boxed{3, 7}

For example, the following question requires multiple answers:
"Find all real solutions of equation $2+7 z+z^2=0$. Does the equation have real solutions? Input Yes or No: [ANS] If your answer is Yes, input the solutions: $z_1=$ [ANS] and $z_2=$ [ANS] with $z_1\le z_2$."
The expected output is \\boxed{True, 6.70156211872, 0.298437881284}

11. Do not abbreviate True/False answers as T/F. Use exactly True or False. e.g. \\boxed{True, False}

"""


SYSTEM_PROMPT_MCQ = r"""
You are an expert competition math assistant.

Your task is to solve multiple choice math problems accurately and efficiently.

Rules:

1. Solve the problem independently BEFORE considering the answer choices.

2. Use concise, rigorous reasoning.
Do NOT ramble, repeat earlier thoughts, or repeatedly reconsider interpretations.

3. If wording is ambiguous:
- choose the most standard mathematical interpretation,
- state the assumption once,
- continue solving.

4. Maintain a short list of:
- known quantities,
- derived equations,
- intermediate conclusions.

5. Derive each fact only once.
Do not restart the entire solution unless a concrete mathematical inconsistency is found.

6. Carefully verify:
- arithmetic,
- algebra,
- signs,
- units,
- substitutions,
before finalizing.

7. After solving:
- compare your computed result against the answer choices,
- eliminate inconsistent options,
- select exactly ONE best answer.

8. Keep reasoning compact and information-dense.
Avoid filler phrases such as:
- "Wait,"
- "Hmm,"
- "another thought,"
- repeated self-questioning.

9. Never change a correct conclusion unless a specific mathematical error is identified.

10. Final output format:
\boxed{LETTER}

Do not output anything after the boxed answer.
"""

def configure_environment(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"


def load_data(data_path):
    with open(data_path, encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]
    return data


def preview_data(data):
    n_mcq = sum(bool(item.get("options")) for item in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    mcq_sample = next((item for item in data if item.get("options")), None)
    free_sample = next((item for item in data if not item.get("options")), None)

    if mcq_sample:
        print("\n-- MCQ sample --")
        print(json.dumps(mcq_sample, indent=2))
    if free_sample:
        print("\n-- Free-form sample --")
        print(json.dumps(free_sample, indent=2))


def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{label}. {option.strip()}" for label, option in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def preview_prompt_samples(data):
    mcq_sample = next((item for item in data if item.get("options")), None)
    free_sample = next((item for item in data if not item.get("options")), None)

    for label, item in (("MCQ", mcq_sample), ("Free-form", free_sample)):
        if not item:
            continue
        _, user_prompt = build_prompt(item["question"], item.get("options"))
        print(f"-- {label} user prompt (first 200 chars) --")
        print(f"{user_prompt[:200]}...\n")


def load_model(model_id, max_tokens):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        presence_penalty=0.0,
        repetition_penalty=1.05,
    )

    return tokenizer, llm, sampling_params


def print_gpu_info():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Capability:", torch.cuda.get_device_capability(0))


def generate_responses(
    data,
    tokenizer,
    llm,
    sampling_params,
):
    prompts = []
    for item in data:
        system_prompt, user_prompt = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]

    for index in range(min(3, len(responses))):
        print(f"\n-- Response {index} (id={data[index].get('id')}) --")
        preview = responses[index][:400]
        suffix = "..." if len(responses[index]) > 400 else ""
        print(f"{preview}{suffix}")

    return responses


def build_unscored_results(data, responses):
    return [
        {
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "response": response,
        }
        for item, response in zip(data, responses)
    ]
    

def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["id", "is_mcq", "response"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            record = {
                "id": result["id"],
                "is_mcq": result["is_mcq"],
                "response": result["response"],
            }
            writer.writerow(record)
            
    print(f"Saved {len(results)} records to {output_path}")


def run_inference():
    configure_environment(GPU_ID)

    data = load_data(DEFAULT_DATA_PATH)
    preview_data(data)
    preview_prompt_samples(data)

    print_gpu_info()
    tokenizer, llm, sampling_params = load_model(MODEL_ID, MAX_TOKENS)
    print("Model loaded.")

    responses = generate_responses(data, tokenizer, llm, sampling_params)
    
    results = build_unscored_results(data, responses)

    save_results(results, DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    run_inference()
