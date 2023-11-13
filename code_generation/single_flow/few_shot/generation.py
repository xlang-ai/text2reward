import re
import time
from typing import List, Dict, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from code_generation.single_flow.classlike_prompt.few_shot_prompt import FEW_SHOT_PROMPT
from code_generation.post_process.post_process import RewardFunctionConverter
from code_generation.single_flow.zero_shot.generation import HuggingFaceLLM


class FewShotGenerator:
    def __init__(self, info_prompt, model_name: str = "gpt-4", examples: List[Dict] = None, k_examples: int = 3,
                 **kwargs: Dict) -> None:
        self.info_prompt = info_prompt

        if model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-0314", "gpt-4-0613"]:
            self.llm = ChatOpenAI(model_name=model_name, **kwargs)
        elif model_name in ["codellama_34b", "llama_2_70b"]:
            self.llm = HuggingFaceLLM(name=model_name, **kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported!")

        self.examples = examples if examples else []
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(),
            Chroma,
            k=k_examples,
        )

    def generate_code(self, instruction: str, map_dict: dict) -> Tuple[str, str]:
        chain = LLMChain(
            prompt=FewShotPromptTemplate(
                example_selector=self.example_selector,
                example_prompt=FEW_SHOT_PROMPT,
                prefix=self.info_prompt,
                suffix="Tasks to be fulfilled: {instruction}",
                input_variables=["instruction"]
            ),
            llm=self.llm
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
