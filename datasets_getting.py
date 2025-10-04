from datasets import load_dataset

"""dataset = load_dataset(
    "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
    split="train[:1000]",
    verification_mode="no_checks"
)"""

dataset = load_dataset(
    "julien31/soar_arc_train_5M",
    split="train[:1000]",
    verification_mode="no_checks"
)