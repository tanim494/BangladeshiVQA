"""Closed-source VQA evaluation over the OpenRouter API (zero-shot and few-shot), following Appendix A.3."""

import os
import csv
import json
import time
import base64
import argparse

from openai import OpenAI
from tqdm import tqdm

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Set OPENROUTER_API_KEY in your environment (see .env.example).")

MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.5-flash")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

MAX_RETRIES = 8

ZERO_SHOT_INSTRUCTION = (
    "তুমি একটি ভিজ্যুয়াল কোশ্চেন আনসারিং (VQA) ডেটাসেট ইভালুয়েটর।\n"
    "নির্দেশনা: ছবি দেখে নিচের প্রশ্নটির সঠিক উত্তরটি সরাসরি লিখবে। উত্তরের আগে বা পরে কোনো অতিরিক্ত কিছু দিবে না।"
)

FEW_SHOT_INSTRUCTION = (
    "তুমি একটি ভিজ্যুয়াল কোশ্চেন আনসারিং (VQA) ডেটাসেট ইভালুয়েটর।\n"
    "নির্দেশনা: ছবি দেখে নিচের প্রশ্নটির সঠিক উত্তরটি সরাসরি লিখবে। উত্তরের আগে বা পরে অতিরিক্ত কিছু দিবে না। নিচের উদাহরণগুলো অনুসরণ করো।"
)

FEW_SHOT_EXAMPLES = [
    {"numeral": "১", "image_id": "CS_1785", "question": "লাল কাঠের কাঠামোর ওপর বাংলায় কী লেখা আছে?", "answer": "চারুকলা রং উৎসব"},
    {"numeral": "২", "image_id": "CS_0502", "question": "জেলেদের কাঁধে কী আছে?", "answer": "বাঁশের পোলো"},
    {"numeral": "৩", "image_id": "CS_0648", "question": "ভাতের ওপরে কয়টি ভাজা ইলিশ মাছের টুকরো রাখা আছে?", "answer": "২ টি"},
]


def text_part(text):
    return {"type": "text", "text": text}


def image_part(image_dir, image_id):
    path = os.path.join(image_dir, f"{image_id}.jpg")
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def build_messages(mode, image_dir, target_image_id, target_question):
    if mode == "few-shot":
        instruction = FEW_SHOT_INSTRUCTION
        content = []
        for example in FEW_SHOT_EXAMPLES:
            content.append(text_part(f"--- উদাহরণ {example['numeral']} ---"))
            content.append(image_part(image_dir, example["image_id"]))
            content.append(text_part(f"প্রশ্ন: {example['question']}"))
            content.append(text_part(f"উত্তর: {example['answer']}"))
        content.append(text_part("--- এখন এই নতুন ছবিটির জন্য প্রশ্নটির উত্তর দাও ---"))
        content.append(image_part(image_dir, target_image_id))
        content.append(text_part(f"প্রশ্ন: {target_question}"))
        content.append(text_part("উত্তর:"))
    else:
        instruction = ZERO_SHOT_INSTRUCTION
        content = [
            text_part("এই নতুন ছবিটির জন্য প্রশ্নটির উত্তর দাও"),
            image_part(image_dir, target_image_id),
            text_part(f"প্রশ্ন: {target_question}"),
            text_part("উত্তর:"),
        ]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]


def get_model_answer(messages):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(model=MODEL, messages=messages)
            answer = response.choices[0].message.content
            if answer and answer.strip():
                return answer.strip()
            raise ValueError("Empty response")
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return ""
            time.sleep(min(2 ** attempt, 60))
    return ""


def load_completed(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as existing:
            for row in csv.DictReader(existing):
                done.add((row["image_id"], row["question"]))
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a closed-source model on BangladeshiVQA via OpenRouter."
    )
    parser.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--test", default="data/annotations/splits/test.json")
    parser.add_argument("--images", default="images")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or f"predictions_{args.mode}_{MODEL.replace('/', '_')}.csv"

    with open(args.test, encoding="utf-8") as test_file:
        test_data = json.load(test_file)

    done = load_completed(output_path)
    fieldnames = ["image_id", "q_type", "question", "ground_truth", "prediction"]
    write_header = not os.path.exists(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for item in tqdm(test_data, desc=f"{MODEL} [{args.mode}]"):
            if (item["image_id"], item["question"]) in done:
                continue
            messages = build_messages(args.mode, args.images, item["image_id"], item["question"])
            prediction = get_model_answer(messages)
            writer.writerow({
                "image_id": item["image_id"],
                "q_type": item.get("q_type", ""),
                "question": item["question"],
                "ground_truth": item.get("answer", ""),
                "prediction": prediction,
            })
            output_file.flush()


if __name__ == "__main__":
    main()
