import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Union

from pydantic import BaseModel, Extra


class Role(str, Enum):
    CONTEXT = "context"
    USER = "user"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    text: str


class FewShot(BaseModel):
    messages: List[Message]


class PromptTemplate(BaseModel, ABC):
    instruction: str
    placeholder_variables: List[str] = []
    raw_prompt: bool = False


class StringPromptTemplate(PromptTemplate, extra=Extra.forbid):

    def to_chat_prompt(self, placeholder_values: Dict[str, str]) -> List[dict]:
        instruction_string = self._replace_placeholders(
            self.instruction, placeholder_values
        )
        return [{"role": "user", "content": instruction_string}]

    def to_prompt_string(self, placeholder_values: Dict[str, str]) -> str:
        return self._replace_placeholders(self.instruction, placeholder_values)

    def to_aleph_alpha_instruction_prompt(
        self, placeholder_values: Dict[str, str]
    ) -> str:
        instruction = self._replace_placeholders(self.instruction, placeholder_values)
        if self.raw_prompt:
            return instruction
        else:
            return f"""### Instruction:
{instruction}

### Response:
"""

    def _replace_placeholders(self, template: str, placeholder_values: dict):
        return template.format(**placeholder_values)


class ChatPromptTemplate(PromptTemplate, extra=Extra.forbid):
    few_shot_separator: str = "\n###\n\n"
    few_shot_enumerator: str = "Dialog {i}:\n"  # Note: can be overwritten; if {i} is not given, then there is simply no enumeration
    user_delimiter: str = "User"
    system_delimiter: str = "System"
    context_delimiter: str = "Context"
    few_shot_as_system: bool = True

    few_shots: List[FewShot]
    template: List[Message]

    def to_chat_prompt(
        self, placeholder_values: Dict[str, str], history: List[Message]
    ) -> List[dict]:
        chat_messages = []
        chat_messages.append(
            {
                "role": "system",
                "content": self._replace_placeholders(
                    self.instruction, placeholder_values
                ),
            }
        )
        chat_messages += self.__get_chat_few_shot_samples()
        chat_messages += self.__get_chat_history(history)
        chat_messages += self.__get_chat_filled_template(placeholder_values)
        return chat_messages

    def __get_chat_few_shot_samples(self):
        samples = []
        for few_shot in self.few_shots:
            for message in few_shot.messages:
                samples.append(
                    {
                        "role": "system" if self.few_shot_as_system else self.__get_chat_delimiter(message),  # TODO: for few shots in langchain they hard code that as "system". But not in OpenAI Doc: https://platform.openai.com/docs/guides/prompt-engineering/tactic-provide-examples
                        "content": message.text,
                        "name": f"example_{message.role.value}",
                    }
                )
        return samples

    def __get_chat_history(self, history: List[Message]) -> List[dict]:
        history_samples = []
        for message in history:
            history_samples.append(
                {"role": self.__get_chat_delimiter(message), "content": message.text}
            )
        return history_samples

    def __get_chat_filled_template(self, placeholder_values: Dict[str, str]):
        template_samples = []
        for message in self.template:
            template_samples.append(
                {
                    "role": self.__get_chat_delimiter(message),
                    "content": self._replace_placeholders(
                        message.text, placeholder_values
                    ),
                }
            )
        if len(template_samples) > 0 and template_samples[-1]["role"] == "assistant":
            return template_samples[:-1]

        return template_samples

    def __get_chat_delimiter(self, message: Message):
        if message.role == Role.USER:
            return "user"
        elif message.role == Role.SYSTEM:
            return "assistant"
        elif message.role == Role.CONTEXT:
            return "system"
        else:
            raise ValueError(f"Unknown role '{message.role}' in prompt template")

    def to_aleph_alpha_instruction_prompt(
        self, placeholder_values: Dict[str, str], history: List[Message]
    ) -> str:
        instruction = self._replace_placeholders(self.instruction, placeholder_values)
        input_string = ""
        if len(self.few_shots) > 0:
            input_string += f"{self.__create_few_shot_string()}\n"
        if len(history) > 0:
            input_string += self.few_shot_separator
            input_string += f"{self.__get_history_string(history)}"
        if len(input_string) > 0:
            input_string += self.few_shot_separator
            # input_string += "\n\n"
        input_string += self.__get_template_string(placeholder_values)
        aleph_alpha_control_prompt = f"""### Instruction:
{instruction}


### Input:
{input_string}


### Response:

"""
        aleph_alpha_control_prompt = re.sub(
            r"\n\n\n", "\n\n", aleph_alpha_control_prompt
        )
        return aleph_alpha_control_prompt

    def to_prompt_string(
        self, placeholder_values: Dict[str, str], history: List[Message]
    ) -> str:
        final_string = self._replace_placeholders(self.instruction, placeholder_values)
        final_string = f"{final_string}\n{self.few_shot_separator}"
        if len(self.few_shots) > 0:
            few_shot_string = self.__create_few_shot_string()
            final_string = (
                f"{final_string}{few_shot_string}\n\n{self.few_shot_separator}\n\n"
            )
        if len(history) > 0:
            history_string = self.__get_history_string(history)
            final_string = (
                f"{final_string}{history_string}\n\n{self.few_shot_separator}\n\n"
            )
        template_string = self.__get_template_string(placeholder_values)
        final_string = f"{final_string}{template_string}"
        final_string = re.sub(r"\n\n\n", "\n", final_string)
        return final_string

    def __create_few_shot_string(self):
        few_shot_string = ""
        for idx, few_shot_sample in enumerate(self.few_shots):
            few_shot_enumerator = self.few_shot_enumerator.format(i=idx + 1)
            few_shot_string = f"{few_shot_string}{few_shot_enumerator}"
            for message in few_shot_sample.messages:
                delimiter = self.__get_delimiter(message)
                if message.role == Role.CONTEXT:
                    few_shot_string = (
                        f"{few_shot_string}{delimiter}:\n{message.text}\n"
                    )
                else:
                    few_shot_string = f"{few_shot_string}{delimiter}: {message.text}\n"
            if (idx+1) < len(self.few_shots):
                few_shot_string = f"{few_shot_string}{self.few_shot_separator}"
        return f"{few_shot_string}"

    def __get_history_string(self, history: List[Message]):
        history_string = ""
        for message in history:
            delimiter = self.__get_delimiter(message)
            if message.role == Role.CONTEXT:
                message_string = f"{delimiter}:\n{message.text}"
            else:
                message_string = f"{delimiter}: {message.text}"
            history_string = f"{history_string}{message_string}\n"
        return f"{history_string}\n\n"

    def __get_template_string(self, placeholder_values: Dict[str, str]):
        template_string = ""
        for message in self.template:
            delimiter = self.__get_delimiter(message)
            message_string = self._replace_placeholders(
                message.text, placeholder_values
            )
            if message.role == Role.CONTEXT:
                message_string = f"{delimiter}:\n{message_string}"
            else:
                message_string = f"{delimiter}: {message_string}"
            template_string = f"{template_string}{message_string}\n"
        return template_string

    def _replace_placeholders(self, template: str, placeholder_values: dict):
        return template.format(**placeholder_values)

    def __get_delimiter(self, message: Message):
        if message.role == Role.USER:
            return self.user_delimiter
        elif message.role == Role.SYSTEM:
            return self.system_delimiter
        elif message.role == Role.CONTEXT:
            return self.context_delimiter
        else:
            raise ValueError(f"Unknown role '{message.role}' in prompt template")


class Prompt(BaseModel):
    placeholder_values: Dict[str, str]
    history: Union[List[str], List[Message]]
    prompt_template: Union[ChatPromptTemplate, StringPromptTemplate]

    def to_chat_prompt(self) -> List[dict]:
        if isinstance(self.prompt_template, ChatPromptTemplate):
            return self.prompt_template.to_chat_prompt(
                self.placeholder_values, self.history
            )
        elif isinstance(self.prompt_template, StringPromptTemplate):
            return self.prompt_template.to_chat_prompt(self.placeholder_values)

    def to_prompt_string(self) -> str:
        if isinstance(self.prompt_template, ChatPromptTemplate):
            return self.prompt_template.to_prompt_string(
                self.placeholder_values, self.history
            )
        elif isinstance(self.prompt_template, StringPromptTemplate):
            return self.prompt_template.to_prompt_string(self.placeholder_values)

    def to_aleph_alpha_instruction_prompt(self) -> str:
        if isinstance(self.prompt_template, ChatPromptTemplate):
            return self.prompt_template.to_aleph_alpha_instruction_prompt(
                self.placeholder_values, self.history
            )
        elif isinstance(self.prompt_template, StringPromptTemplate):
            return self.prompt_template.to_aleph_alpha_instruction_prompt(
                self.placeholder_values
            )
