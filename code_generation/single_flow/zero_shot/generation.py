import re
import time
from typing import Any, List, Mapping, Optional, Tuple

import torch
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from code_generation.post_process.post_process import RewardFunctionConverter


class HuggingFaceLLM(LLM):
    name: str
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return self.name

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        name_map = {
            "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
            "llama_2_70b": "meta-llama/Llama-2-70b-chat-hf"
        }
        assert self.name in name_map, f"Model name {self.name} not supported!"
        model = name_map[self.name]
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipe = pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        pipe.tokenizer.pad_token_id = tokenizer.eos_token_id

        chat = [
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(chat, tokenize=False)

        raw_results = pipe(
            [prompt],
            do_sample=False,
            # temperature=self.temperature,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
            batch_size=1
        )
        print(raw_results[0][0]["generated_text"][len(prompt):])
        return raw_results[0][0]["generated_text"][len(prompt):]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"name": self.name, "temperature": self.temperature}


class ZeroShotGenerator:
    def __init__(self, info_prompt: PromptTemplate, model_name="gpt-4", **kwargs) -> None:
        if model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-0314", "gpt-4-0613"]:
            self.chain = LLMChain(prompt=info_prompt, llm=ChatOpenAI(model_name=model_name, **kwargs))
        elif model_name in ["codellama_34b", "llama_2_70b"]:
            self.chain = LLMChain(prompt=info_prompt, llm=HuggingFaceLLM(name=model_name, **kwargs))
        else:
            raise ValueError(f"Model name {model_name} not supported!")

    def generate_code(self, instruction: str, map_dict: dict) -> Tuple[str, str]:
        code_content = ""
        while True:
            response = self.chain.run(**{"instruction": instruction})
            pattern = r"\```python\n(.+?)\n```" if "```python" in response else r"\```\n(.+?)\n```"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code_content = match.group(1)
                break
            else:
                print(response)
                time.sleep(5)
                print("No match!")
                continue

        general_code = code_content

        # Post-processing, replace the general terms with specific terms
        converter = RewardFunctionConverter(map_dict)
        specific_code = converter.general_to_specific(general_code)

        return general_code, specific_code
