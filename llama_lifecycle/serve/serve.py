from transformers import AutoTokenizer, LlamaForCausalLM


class LlamaChatbot:
    def __init__(
        self,
        model_path: str = "../../models/llama-2-7b-chat-sft/qlora_adapter",
        tokenizer_path: str = "../../models/llama-2-7b-chat-hf",
        device_map: str | None = "cuda:0",
        max_memory: str | dict | None = "1200MB",
    ):
        self.device_map = device_map
        self.max_memory = max_memory

        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map=device_map,
            max_memory=max_memory,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path=tokenizer_path,
            device_map=device_map,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, max_length: int = 300):
        def formatting_func(example):
            text = f"\n ### USER: {example}"
            return text

        inputs = self.tokenizer(
            formatting_func(prompt), return_tensors="pt"
        ).input_ids.to(self.max_memory)
        generate_ids = self.model.generate(
            inputs=inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(
            generate_ids=generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
