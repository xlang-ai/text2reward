import os
import re
import time
from typing import Dict, List, Tuple

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from code_generation.interactive.classlike_prompt.feedback_prompt import FEEDBACK_PROMPT

from code_generation.post_process.post_process import RewardFunctionConverter


class BasicHumanFeedbackCodeGenerator:
    def __init__(self, info_prompt: PromptTemplate, model_name: str = "gpt4", **kwargs: Dict) -> None:
        self.info_prompt = info_prompt
        self.llm = AzureChatOpenAI(
            openai_api_base="",
            openai_api_version="2023-05-15",
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            openai_api_type="azure",
            deployment_name=model_name,
            temperature=0.7,
            request_timeout=120,
            max_retries=5,
            streaming=False,
            verbose=True
        )

    def generate_code(self, instruction: str, map_dict: dict, history: List[Dict]) -> Tuple[str, str]:
        chain = LLMChain(
            prompt=FewShotPromptTemplate(
                examples=history,
                example_prompt=FEEDBACK_PROMPT,
                prefix=self.info_prompt.format(instruction=instruction),
                suffix="Re-imagine which steps is missed or wrong.\nShow me the improved code as below:",
                input_variables=[]
            ),
            llm=self.llm,
            verbose=True
        )

        code_content = ""
        while True:
            response = chain.run(**{"instruction": instruction})
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
