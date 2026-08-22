import os
import time
import pandas as pd
from google import genai
from google.genai import types

# -------- Clear Old Path Cache --------
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
LOCATION = "global"
MODEL_ID = "gemini-3.6-flash"

SAVE_MILESTONE = 20

# 🎯 Local PC paths
INPUT_CSV_PATH = "/home/user/Documents/Study/Research/Own/BVQA Dataset/Local PC/Results/On Test Split/SmolVl-2-2.2B/smolvlm2_fineTuned_test.csv"
OUTPUT_CSV_PATH = "/home/user/Documents/Study/Research/Own/BVQA Dataset/Local PC/Results/On Test Split/SmolVl-2-2.2B/smolvlm2_fineTuned_validated_by_llm.csv"

# Initialize Vertex AI Client
client = genai.Client(
    project=PROJECT_ID,
    location=LOCATION,
    vertexai=True
)


def get_llm_judgment(question, ground_truth, prediction):
    max_retries = 8
    backoff_factor = 2

    system_instruction = (
        "তুমি একজন বিশেষজ্ঞ ভাষাবিদ এবং Visual Question Answering (VQA) ইভ্যালুয়েটর। "
        "তোমার কাজ হলো একটি মডেলের 'Prediction' এবং আসল 'Ground Truth' উত্তরটি অর্থের দিক থেকে একই কিনা তা যাচাই করা।\n\n"
        "নিয়মাবলী:\n"
        "১. বানানের সামান্য ভিন্নতা, সমার্থক শব্দ (synonyms), বা বাক্যাংশের ভিন্নতা থাকলে এবং মূল অর্থ ঠিক থাকলে সেটিকে সঠিক হিসেবে ধরবে।\n"
        "২. সংখ্যা এবং সংখ্যার বানান (যেমন '১' এবং 'এক') একই অর্থ বহন করে।\n"
        "৩. বাংলা নির্দেশক (যেমন: টি, টা, জন) কম বা বেশি থাকলে অর্থের পরিবর্তন না হলে তা সঠিক।\n"
        "৪. Ground Truth যদি শুধুমাত্র 'হ্যাঁ' বা 'না' হয়, তবে Prediction-কেও স্পষ্টভাবে একই অর্থ প্রকাশ করতে হবে।\n\n"
        "আউটপুট ফরম্যাট:\n"
        "কোনো ব্যাখ্যা, প্রিফিক্স বা অতিরিক্ত শব্দ ছাড়া শুধুমাত্র '1' (যদি সঠিক হয়) অথবা '0' (যদি ভুল হয়) লিখবে।"
    )

    prompt = (
        f"প্রশ্ন: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Prediction: {prediction}\n\n"
        "উত্তর (1/0):"
    )

    contents = [system_instruction, prompt]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )

            time.sleep(0.5)

            return response.text.strip()

        except Exception as e:
            error_msg = str(e)

            if (
                "429" in error_msg
                or "Quota" in error_msg
                or "ResourceExhausted" in error_msg
            ):
                sleep_time = backoff_factor ** (attempt + 1)
                print(
                    f"⚠️ Rate limit hit! Waiting {sleep_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)
            else:
                return f"ERROR: {error_msg}"

    return "ERROR: Rate limit exceeded."


def extract_verdict(llm_response):
    """Extract only 1 or 0 from model response."""

    try:
        value = int(llm_response)
        if value in [0, 1]:
            return value
    except Exception:
        pass

    if "1" in llm_response:
        return 1
    elif "0" in llm_response:
        return 0

    return None


def main():

    print("🚀 Starting LLM Validation")

    # --------------------------------------------------
    # Resume from previous output if it exists
    # --------------------------------------------------
    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"📂 Resuming from existing file:\n{OUTPUT_CSV_PATH}")
        df = pd.read_csv(OUTPUT_CSV_PATH, encoding="utf-8-sig")
    else:
        print(f"📂 Starting new evaluation:\n{INPUT_CSV_PATH}")
        df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8-sig")

        if "llm_verdict" not in df.columns:
            df["llm_verdict"] = pd.NA

    total_rows = len(df)

    # Count already completed rows
    processed_count = df["llm_verdict"].notna().sum()

    print(f"Already processed: {processed_count}/{total_rows}")
    print("=" * 60)

    for idx, row in df.iterrows():

        # Skip completed rows
        if pd.notna(row["llm_verdict"]):
            continue

        question = row["question"]

        if "ground_truth" in df.columns:
            ground_truth = row["ground_truth"]
        else:
            ground_truth = row["answer"]

        prediction = row["prediction"]

        response_text = get_llm_judgment(
            question,
            ground_truth,
            prediction,
        )

        verdict = extract_verdict(response_text)

        df.at[idx, "llm_verdict"] = verdict

        processed_count += 1

        print(
            f"[{processed_count}/{total_rows}] "
            f"LLM: {verdict} | "
            f"GT: {ground_truth} | "
            f"Pred: {prediction}"
        )

        # Save checkpoint
        if (
            processed_count % SAVE_MILESTONE == 0
            or processed_count == total_rows
        ):
            df.to_csv(
                OUTPUT_CSV_PATH,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"💾 Checkpoint saved ({processed_count}/{total_rows})")

    # Final save
    df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n🎉 Evaluation complete!")
    print(f"✅ Output saved to:\n{OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()