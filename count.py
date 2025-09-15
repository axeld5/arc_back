
import tiktoken

# Count tokens for each problem
encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

train_input_tokens = []
train_output_tokens = []
test_input_tokens = []

for conversation in train_problems["conversations"]:
    user_content = conversation[0]["content"]
    assistant_content = conversation[1]["content"]
    
    input_tokens = len(encoding.encode(user_content))
    output_tokens = len(encoding.encode(assistant_content))
    
    train_input_tokens.append(input_tokens)
    train_output_tokens.append(output_tokens)

for conversation in test_problems["conversations"]:
    user_content = conversation[0]["content"]
    input_tokens = len(encoding.encode(user_content))
    test_input_tokens.append(input_tokens)

# Calculate statistics
import statistics

print("=== TOKEN STATISTICS ===")
print(f"Training problems: {len(train_input_tokens)}")
print(f"Test problems: {len(test_input_tokens)}")
print()

print("TRAINING INPUT TOKENS:")
print(f"  Mean: {statistics.mean(train_input_tokens):.1f}")
print(f"  Median: {statistics.median(train_input_tokens):.1f}")
print(f"  Max: {max(train_input_tokens)}")
print(f"  Min: {min(train_input_tokens)}")
print(f"  Std Dev: {statistics.stdev(train_input_tokens):.1f}")
print()

print("TRAINING OUTPUT TOKENS:")
print(f"  Mean: {statistics.mean(train_output_tokens):.1f}")
print(f"  Median: {statistics.median(train_output_tokens):.1f}")
print(f"  Max: {max(train_output_tokens)}")
print(f"  Min: {min(train_output_tokens)}")
print(f"  Std Dev: {statistics.stdev(train_output_tokens):.1f}")
print()

print("TEST INPUT TOKENS:")
print(f"  Mean: {statistics.mean(test_input_tokens):.1f}")
print(f"  Median: {statistics.median(test_input_tokens):.1f}")
print(f"  Max: {max(test_input_tokens)}")
print(f"  Min: {min(test_input_tokens)}")
print(f"  Std Dev: {statistics.stdev(test_input_tokens):.1f}")
print()

print("TOTAL TRAINING TOKENS:")
total_train_tokens = sum(train_input_tokens) + sum(train_output_tokens)
print(f"  Total: {total_train_tokens:,}")
print(f"  Input: {sum(train_input_tokens):,}")
print(f"  Output: {sum(train_output_tokens):,}")