from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch"
)
print("✅ TrainingArguments works!")

