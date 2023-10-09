import json5
import re
import time
from typing import Dict
from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel


class RewardFunctionConverter:
    def __init__(self, map_dict=None) -> None:
        self.map_dict = {k: map_dict[k] for k in
                         sorted(map_dict, key=len, reverse=True)} if map_dict is not None else {}

    def specific_to_general(self, specific_code: str) -> str:
        general_code = specific_code

        # replace specific terms with general terms
        for k, v in self.map_dict.items():
            general_code = general_code.replace(v, k)

        return general_code

    def general_to_specific(self, general_code: str) -> str:

        specific_code = general_code

        # replace general terms with specific terms
        for k, v in self.map_dict.items():
            specific_code = specific_code.replace(k, v)
        return specific_code