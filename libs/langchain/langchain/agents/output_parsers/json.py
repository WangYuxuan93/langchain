from __future__ import annotations

import logging
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import parse_json_markdown

from langchain.agents.agent import AgentOutputParser

logger = logging.getLogger(__name__)

import re
import json

def parse_all_json(text):
    pattern = re.compile(r'{\s*"action":\s*"[^"]+",\s*"action_input":\s*(?:\{.*?\}|\s*".*?"\s*)\s*}', re.DOTALL)

    matches = pattern.findall(text)

    # 替换 True 和 False 为 true 和 false
    def replace_boolean_values(json_str):
        json_str = json_str.replace(' True', ' true')
        json_str = json_str.replace(' False', ' false')
        return json_str

    # 处理所有匹配的部分
    json_scripts = []
    for match in matches:
        valid_json_str = replace_boolean_values(match)
        json_scripts.append(json.loads(valid_json_str))

    return json_scripts

class JSONAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in JSON format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    {
      "action": "search",
      "action_input": "2+2"
    }
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    {
      "action": "Final Answer",
      "action_input": "4"
    }
    ```
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            #print ("json parser origin input:\n",text)
            response = parse_json_markdown(text, parser=parse_all_json)
            #print ("json parser response:\n",response)
            #exit()
            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
                #logger.warning("Got multiple action responses: %s", response)
                first_response = response[0]
                #print ("json parser multi input\n:",response)
            if response[-1]["action"] == "Final Answer":
                return AgentFinish({"output": response}, text)
            else:
                return AgentAction(
                    first_response["action"], first_response.get("action_input", {}), text
                )
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "json-agent"
