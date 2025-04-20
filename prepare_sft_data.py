from datasets import load_dataset
import json
import os

# 1. Load Alpaca-cleaned
dataset = load_dataset("yahma/alpaca-cleaned")

# 2. Peek at one example
print("Sample instruction-response pair:")
print(dataset["train"][0])

# 3. Format each example into a single text field
def format_example(example):
    prompt = (
        f"### Instruction:\n{example['instruction'].strip()}\n\n"
        f"### Response:\n{example['output'].strip()}"
    )
    return {"text": prompt}

# Apply formatting and drop the instruction/input/output columns
dataset = dataset.map(
    format_example,
    remove_columns=["instruction", "input", "output"],
    num_proc=4  # parallelize if you have multiple cores
)

# 4. Split into train/validation
splits = dataset["train"].train_test_split(test_size=0.1, seed=42)

# 5. Save to JSONL files
os.makedirs("data", exist_ok=True)
for split_name, split_data in splits.items():
    path = f"data/{split_name}.jsonl"
    with open(path, "w") as f:
        for ex in split_data:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {split_name} ({len(split_data)}) examples to {path}")
