from pathlib import Path
from typing import Iterator, cast

from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage

from lib.inference import InferenceBase
from lib.types.config import ParamsConfig
from lib.types.other import ChatCompletionResponse, PetAction, RoleContent


class LlamaCpp(InferenceBase):
    """Inference adapter that uses llama-cpp-python"""

    def create_chat_completion(self, messages: list[RoleContent]) -> RoleContent:
        msg_list = cast(list[ChatCompletionRequestMessage], [msg.model_dump() for msg in messages])

        response = self.llm.create_chat_completion(
            msg_list,
            temperature=self.config.temperature,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty,
            repeat_penalty=self.config.repeat_penalty,
            min_p=self.config.min_p,
            seed=self.config.seed,
            response_format={
                "type": "json_object",
                "schema": PetAction.model_json_schema(),  # type:ignore
            },
        )

        assert not isinstance(response, Iterator)
        parsed_response = ChatCompletionResponse(**response)  # type:ignore[arg-type]

        return parsed_response.get_message()

    def __init__(self, config: ParamsConfig, model_path: Path):
        self.config = config
        self.model_path = str(model_path.resolve())
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.config.context_size,
            n_gpu_layers=-1,
            verbose=False,
        )
