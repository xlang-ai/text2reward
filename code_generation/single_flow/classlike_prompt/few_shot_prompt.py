from langchain.prompts import PromptTemplate

few_shot_prompt = """
An example:
Tasks to be fulfilled: {instruction}
Corresponding reward function: 
```python
{reward_code}
```
"""

FEW_SHOT_PROMPT = PromptTemplate(
    input_variables=["instruction", "reward_code"],
    template=few_shot_prompt
)
