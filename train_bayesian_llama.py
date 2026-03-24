import json
import torch
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_FILE = "bayesian_flight_data.jsonl" #training data
MAX_SEQ_LENGTH = 2048 #max token length to save VRAM.
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # we use 4-bit quanticized LLama-3.1. research shows that a 4-bit base model with 16-bit trainable adapters(from LoRA) can achieve 99% of the performance with 16-bit.
OUTPUT_DIR = "llama_lora"

# ==========================================
# 2. DATA LOADING & SPLITTING
# ==========================================
print("loading the dataset...we split the dataset using 90-5-5 split by user for train-val-test")

with open(DATA_FILE, "r") as f:
    raw_lines = [json.loads(line) for line in f] #we create a list of dictionaries, where each dictionary has the format {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# each line in raw_lines corresponds to one interaction, but each user has 4 interactions sequentially. we split the dataset by user and group interactions of the same user together.

INTERACTIONS_PER_USER = 4
users = [raw_lines[i:i + INTERACTIONS_PER_USER] for i in range(0, len(raw_lines), INTERACTIONS_PER_USER)] #group every 4 interactions from a user

num_users = len(users) 
train_idx = int(0.90 * num_users)
val_idx = int(0.95 * num_users)

# Split the users
train_users = users[:train_idx]
val_users = users[train_idx:val_idx]
test_users = users[val_idx:]

# Flatten back into individual interaction lists
train_data = [interaction for user in train_users for interaction in user]
val_data = [interaction for user in val_users for interaction in user]
test_data = [interaction for user in test_users for interaction in user]

print(f"The dataset split is as follows: {len(train_data)} Train | {len(val_data)} Val | {len(test_data)} Test")

# Convert to Hugging Face Dataset objects:
# The data from the list is stored in an Apache Arrow format, which uses memory mapping - doesn't load everything into RAM at once, but transfers from hard drive to GPU only when needed.
# We wrap the dictionary in a DatasetDict structure to adhere to the expected input format for SFT Trainer.
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data) 
})

# ==========================================
# 3. MODEL & TOKENIZER INITIALIZATION
# ==========================================
# We use from_pretrained to load the base model and apply the following adjustments:
# architecture auto-detection
# quantization with regard to LoRA
# replace standard pytorch operations with optimized versions for memory efficincy
# forward pass rewriting
# applies chat templates and aligns tokenization with the training format

print(f"Loading base model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto-detects fp16 or bf16 based on your GPU
    load_in_4bit = True, # Crucial for fitting in standard GPUs
)

# apply LoRA adapters to the base model. We use low rank matrices as a method to tune only a small number of parameters, which allows us to train on a single GPU
# we use r=16 as a solid default for teaching a reasoning task.
# we apply not only to attention layers but also to the feedforward neural network. Achieves maximum flexibility.

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA rank (16 is a solid default for reasoning tasks)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16, 
    lora_dropout = 0, #because we are using unsloth 
    bias = "none", #no bias used with LoRA
    use_gradient_checkpointing = "unsloth", # save VRAM
    random_state = 3407, #Torch.manual_seed(3407) is all you need
)

# ==========================================
# 4. CHAT TEMPLATE FORMATTING
# ==========================================
# Currently we have the data in the format {"role": "...", "content": "..."} JSON and we want to format it into the exact string
# Unsloth provides a way to translate this perfectly in a way that is personalized for LLama-3

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", 
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
)

def format_chat(examples):
    # Applies the chat template to the "messages" column
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
    return {"text": texts}

print("Applying Llama-3 chat templates...")
dataset = dataset.map(format_chat, batched=True)

# ==========================================
# 5. TRAINING LOOP SETUP
# ==========================================

# Llama-3's specific hidden tokens that introduce the assistant's turn
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# Initialize the highly-focused collator
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    data_collator = collator,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Adjust based on VRAM (2-4 is usually safe for 24GB VRAM)
        gradient_accumulation_steps = 4, # Simulates a larger batch size
        warmup_steps = 50,
        max_steps = 500, # Start with 500 steps to see how it learns. For a full run, use `num_train_epochs=1` instead.
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        eval_steps = 50, # Evaluate on the validation set every 50 steps
        evaluation_strategy = "steps",
        optim = "adamw_8bit", # 8-bit optimizer saves more memory
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# ==========================================
# 6. EXECUTE TRAINING
# ==========================================
print("Starting training...")
trainer_stats = trainer.train()

# ==========================================
# 7. SAVE THE MODEL
# ==========================================
print(f"Training complete. Saving LoRA adapters to {OUTPUT_DIR}")
# saves only the LoRA adapters
model.save_pretrained(OUTPUT_DIR) 
tokenizer.save_pretrained(OUTPUT_DIR)

print("saved!")