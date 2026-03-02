from text_preprocessing import lang_id, normalize_query, embedder
from retrieval_chain import retrieve_context
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load fine-tuned model (after training)
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype="auto", device_map="mps"
)
# Load LoRA adapter if available
import os
adapter_path = "./support-agent/final" if os.path.exists("./support-agent/final") else "./support-agent/checkpoint-6"
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"Loaded fine-tuned LoRA adapter from {adapter_path}.")
except Exception as e:
    print(f"No fine-tuned adapter found ({e}). Using base model.")

LANG_NAMES = {
    "en": "English", "sw": "Swahili", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
    "pl": "Polish", "tr": "Turkish", "ru": "Russian", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
}

def respond(user_query: str) -> str:
    # 1. Detect language (default to English if low confidence)
    detected = lang_id(user_query)
    lang = detected[0]["label"]
    confidence = detected[0]["score"]
    if confidence < 0.5:
        lang = "en"
    lang_name = LANG_NAMES.get(lang, lang)
    print(f"[Language: {lang_name} ({confidence:.2f})]")

    # 2. Normalize query
    clean_query = normalize_query(user_query)

    # 3. Retrieve relevant context via RAG
    rag_context = retrieve_context(clean_query)

    # 4. Build prompt with retrieved context
    prompt = (
        f"<s>[INST] You are a helpful multilingual customer support agent. "
        f"The user is writing in {lang_name}. You MUST respond ONLY in {lang_name}.\n\n"
        f"Context from knowledge base:\n{rag_context}\n\n"
        f"User query: {user_query} [/INST]"
    )

    # 5. Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
    # Decode only the newly generated tokens (exclude the input prompt)
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return response


if __name__ == "__main__":
    print("=" * 60)
    print("  Multilingual Customer Support Agent")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        answer = respond(query)
        print(f"\nAgent: {answer}")
