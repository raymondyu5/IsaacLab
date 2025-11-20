import openai
from openai import OpenAI
import os

openai_api_key = os.environ.get("OPENAI_KEY")
import pathlib
import pystache
import re

openai_client = OpenAI(api_key=openai_api_key)
ENGINE = "gpt-3.5-turbo"


class LMPAgent:

    def __init__(self, text_prompt_path, failure_template):
        self.prompt = self.parse_text_prompt(text_prompt_path)
        self.user_prompt = self.parse_text_prompt(failure_template)

    def parse_text_prompt(self, text_prompt_path):

        with open(text_prompt_path, "r") as file:
            template = file.read()

        return template

    def prompt_llm(self, language_instruction, failure_reasoning,
                   previous_failure_reasoning, previous_llm_feedback):

        user_prompt = pystache.render(
            self.user_prompt, {
                "language_instruction": language_instruction,
                "failure_reasoning": failure_reasoning,
                "previous_failure_reasoning": previous_failure_reasoning,
                "previous_llm_feedback": previous_llm_feedback
            })

        messages = [{
            "role": "system",
            "content": self.prompt
        }, {
            "role": "user",
            "content": user_prompt
        }]

        completion = openai_client.chat.completions.create(model=ENGINE,
                                                           messages=messages,
                                                           temperature=0)

        return self.parse_code(str(completion.choices[0].message.content))

    def parse_code(self, input_text):
        pattern = "```python(.*?)```"
        matches = re.findall(pattern, input_text, re.DOTALL)
        if len(matches) == 0:
            return None

        all_code = ""
        for match in matches:
            all_code += "\n" + match
        return all_code
