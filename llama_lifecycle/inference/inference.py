from transformers import AutoTokenizer, LlamaForCausalLM


class LlamaChatbot:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device_map: str | None,
        max_memory: str | dict | None,
    ):
        self.device_map = device_map
        self.max_memory = max_memory
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            device_map=device_map,
            max_memory=max_memory,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
            device_map=device_map,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, max_length: int = 300):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.device_map)
        generate_ids = self.model.generate(
            inputs=inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.batch_decode(
            sequences=generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
