from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "../../models/llama-2-7b-chat-sft/qlora_adapter",
    device_map="cuda:0",
    max_memory="12000MB",
)
tokenizer = AutoTokenizer.from_pretrained(
    "../../models/llama-2-7b-chat-hf", device_map="cuda:0"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Generate
generate_ids = model.generate(
    inputs, max_length=300, pad_token_id=tokenizer.pad_token_id
)
tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
