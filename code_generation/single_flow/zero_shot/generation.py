import re
import time
from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from code_generation.post_process.post_process import RewardFunctionConverter


class ZeroShotGenerator:
    def __init__(self, info_prompt: PromptTemplate, model_name="gpt-4", **kwargs) -> None:
        self.chain = LLMChain(prompt=info_prompt, llm=ChatOpenAI(model_name=model_name, **kwargs))

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
