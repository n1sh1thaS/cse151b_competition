#!/usr/bin/env python3

import json
import os
import re
import random

from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID = "0"
MAX_TOKENS = 32768

DEFAULT_DATA_PATH = "data/public.jsonl"
DEFAULT_OUTPUT_PATH = "results/starter_results.jsonl"

SAVE_EVAL = True
DATA_LIMIT = None


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

8. Fully simplify the final result.

9. Never change a correct conclusion unless a specific mathematical error is found.

10. Final output format:
\boxed{final answer}
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


def load_data(data_path, limit=None):
    with open(data_path, encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]
    if limit is not None:
        data = data[:limit]
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
        gpu_memory_utilization=0.50,
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
    RESPONSE_BUFFER_PATH = 'response_buffer.jsonl'
    target_data = data[:200]
    
    responses = []
    chunk_size = 5
    
    for i in range(0, len(target_data), chunk_size):
        chunk_items = target_data[i : i + chunk_size]
        prompts = []
        
        # Build prompts for the current chunk
        for item in chunk_items:
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

        print(f"\n--- Processing questions {i + 1}-{i + len(prompts)} (Chunk {i // chunk_size + 1}) ---")        
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        chunk_responses = [output.outputs[0].text.strip() for output in outputs]
        
        # Keep track of everything for the final return
        responses.extend(chunk_responses)

        # 3. Save these 5 responses to the buffer immediately
        with open(RESPONSE_BUFFER_PATH, "a", encoding="utf-8") as f:
            for r in chunk_responses:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(chunk_responses)} responses to {RESPONSE_BUFFER_PATH}")

        # Optional: Preview the responses from this specific chunk
        for index in range(len(chunk_responses)):
            print(f" -> Response Preview (id={chunk_items[index].get('id')}):")
            preview = chunk_responses[index][:150]
            suffix = "..." if len(chunk_responses[index]) > 150 else ""
            print(f"    {preview}{suffix}")

    return responses


def extract_letter(text):
    boxed_match = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if boxed_match:
        return boxed_match.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response, gold_letter):
    return extract_letter(response) == gold_letter.strip().upper()


def score_results(data, responses):
    from judger import Judger

    judger = Judger(strict_extract=False)
    results = []

    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append(
            {
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "gold": gold,
                "response": response,
                "correct": correct,
            }
        )

    print(f"Scoring complete. {len(results)} results.")
    return results


def build_unscored_results(data, responses):
    return [
        {
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "response": response,
        }
        for item, response in zip(data, responses)
    ]


def print_metrics(results):
    mcq_results = [result for result in results if result["is_mcq"]]
    free_results = [result for result in results if not result["is_mcq"]]

    def accuracy(subset):
        if not subset:
            return 0.0
        return sum(result["correct"] for result in subset) / len(subset) * 100

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(
        f"  MCQ        : {sum(r['correct'] for r in mcq_results):4d} / "
        f"{len(mcq_results):4d}  ({accuracy(mcq_results):.2f}%)"
    )
    print(
        f"  Free-form  : {sum(r['correct'] for r in free_results):4d} / "
        f"{len(free_results):4d}  ({accuracy(free_results):.2f}%)"
    )
    print(
        f"  Overall    : {sum(r['correct'] for r in results):4d} / "
        f"{len(results):4d}  ({accuracy(results):.2f}%)"
    )
    print("=" * 50)


def save_results(results, output_path, save_eval):
    with open(output_path, "w", encoding="utf-8") as handle:
        for result in results:
            if save_eval:
                record = {
                    "id": result["id"],
                    "is_mcq": result["is_mcq"],
                    "gold": result["gold"],
                    "response": result["response"],
                    "correct": result["correct"],
                }
            else:
                record = {
                    "id": result["id"],
                    "is_mcq": result["is_mcq"],
                    "response": result["response"],
                }
            handle.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} records to {output_path}")


def main() -> None:
    configure_environment(GPU_ID)

    data = load_data(DEFAULT_DATA_PATH, limit=DATA_LIMIT)
    preview_data(data)
    preview_prompt_samples(data)

    print_gpu_info()
    tokenizer, llm, sampling_params = load_model(MODEL_ID, MAX_TOKENS)
    print("Model loaded.")

    responses = generate_responses(data, tokenizer, llm, sampling_params)

    save_eval = SAVE_EVAL and all("answer" in item for item in data)
    if save_eval:
        results = score_results(data, responses)
        print_metrics(results)
    else:
        print("Skipping evaluation because answers are unavailable or --no-save-eval was set.")
        results = build_unscored_results(data, responses)

    save_results(results, DEFAULT_OUTPUT_PATH, save_eval=save_eval)


if __name__ == "__main__":
    main()
