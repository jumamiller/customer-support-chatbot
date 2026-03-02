from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="mps")
#lora
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# Load and tokenize dataset
dataset = load_dataset("json", data_files="./data/train.json")["train"]

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

trainer=Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./support-agent",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="epoch",
    ),
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()

# Save final adapter
model.save_pretrained("./support-agent/final")
tokenizer.save_pretrained("./support-agent/final")
print("Final adapter saved to ./support-agent/final/")