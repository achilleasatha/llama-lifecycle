from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "../../models/llama-2-7b-chat-sft/qlora_adapter",
    device_map="cuda:0",
    max_memory="12000MB",
)
tokenizer = AutoTokenizer.from_pretrained("../../models/llama-2-7b-chat-hf")


text = (
    "### USER: Can you explain contrastive learning in machine learning in simple terms for "
    "someone new to the field of ML?### Assistant:"
)

inputs = tokenizer(text, return_tensors="pt").to(0)
outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

print("After attaching Lora adapters:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

model.disable_adapters()
outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

print("Before Lora:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
