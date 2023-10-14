#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import asyncio
import re
from typing import List
from functools import reduce
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, \
                         StoppingCriteria, StoppingCriteriaList, \
                         pipeline, GenerationConfig, TextGenerationPipeline

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class ChatTuple:
    """
    Grouped chat record for rendering prompt.
    """
    system:str = None
    user:str = None
    bot:str = None

    @classmethod
    def from_chat_history(cls, chat_history: [BaseMessage]):
        """
        Convert a list of BaseMessages to a list o ChatTuple.
        Noticed that it's expected that a bot message is an immediate successor of a user message.
        """
        chat_tuples = []
        last_chat_tuple = ChatTuple()
        def commit():
            nonlocal chat_tuples, last_chat_tuple
            chat_tuples.append(last_chat_tuple)
            last_chat_tuple = ChatTuple()

        for chat in chat_history:
            match chat:
                case SystemMessage():
                    last_chat_tuple.system = chat.content
                case HumanMessage():
                    if last_chat_tuple.user != None: commit()
                    last_chat_tuple.user = chat.content
                case AIMessage():
                    assert last_chat_tuple.user != None
                    last_chat_tuple.bot = chat.content
                    commit()
        if last_chat_tuple != ChatTuple(): commit()
        return chat_tuples

TAIDE_PROMPT_TEMPLATE = """{% for chat in history %}
<s>[INST] {% if chat.system is not none %}<<SYS>>
{{chat.system}}
<</SYS>>

{% endif %}{{chat.user}} [/INST]
 {% if chat.bot is not none %}{{chat.bot}} </s>{% endif %}{% endfor %}"""

TAIDE_PROMPT = PromptTemplate.from_template(
    input_type={'history': [ChatTuple]},
    template=TAIDE_PROMPT_TEMPLATE,
    template_format='jinja2'
)

# Prepend this list to the chat history can prevent the model from output English.
DEFAULT_HISTORY_PREFIX = [
    SystemMessage(content='You are a helpful assistant. 你是一個樂於助人的助手。'),
    HumanMessage(content='請用中文回答我'),
    AIMessage(content='當然!為方便溝通,我使用的是傳統中文語言。您有何請求或疑問,請慷慨請教我?'),
]

class StopOnTokens(StoppingCriteria):
    def __init__(self,
                 tokenizer:AutoTokenizer,
                 stop_list:list = ['[INST]', '\nQuestion:', '[INST: ]']): 

        to_token_id = lambda x: torch.LongTensor(tokenizer(x)['input_ids']).to('cuda')
        self.stop_token_ids = map(to_token_id, stop_list)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.all(input_ids[0][-len(stop_ids):] == stop_ids):
                return True
        return False

class TaideChatModel(SimpleChatModel):

    @property
    def _llm_type(self) -> str:
        return "taide-chat-model"

    input_token_limit:int = 3500
    prepend_system_prompt:bool = True
    tokenizer:AutoTokenizer = None
    model:AutoModelForCausalLM = None
    text_generation_pipe:TextGenerationPipeline = None

    def load_model(self, model_path = '/var/models/llama2-7b-chat-b5.0.0'):

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=AutoConfig.from_pretrained(model_path),
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()

        self.text_generation_pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task='text-generation',
            stopping_criteria=StoppingCriteriaList([StopOnTokens(self.tokenizer)]),
            max_length=4096,
            # max_new_tokens=2048,
            # num_beams=2, early_stopping=True, # Beam search
            do_sample=True, temperature=0.2, top_p=0.95, # Top-p (nucleus) sampling
            # penalty_alpha=0.6, top_k=3, low_memory=True, # Contrastive search
            repetition_penalty=1.0,
        )

    def is_too_long(self, messages: [BaseMessage]) -> bool:
        """
        Estimate whether the prompt generated by the given chat history will be too long.
        This public API can be use to evaluate how many documents can be placed in the context window.
        """
        
        chat_history = ChatTuple.from_chat_history(messages)
        system_prompt = ChatTuple.from_chat_history(DEFAULT_HISTORY_PREFIX)
        prompt = TAIDE_PROMPT.format(
                    history=(system_prompt if self.prepend_system_prompt else [])+chat_history
                    )
        return self._is_too_long(prompt)[0]
    
    def _is_too_long(self, sentence: str) -> bool:
        """
        Calculate whether the number of tokens of given sentence exceeds the threshold.
        """

        num_tokens = len(self.tokenizer.tokenize(sentence, add_special_tokens=False))
        return num_tokens >= self.input_token_limit, num_tokens

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a message from given messages.
        """
        
        result = ''
        try:
            
            stopping_criteria = []
            if stop != None:
                stopping_criteria.append(StopOnTokens(tokenizer=self.tokenizer, stop_list=stop))
            system_prompt = ChatTuple.from_chat_history(DEFAULT_HISTORY_PREFIX)
            chat_history = ChatTuple.from_chat_history(messages)
            
            # Trim the over-length history
            prompt = ''
            prompt_tokens = 0
            while True:
                prompt = TAIDE_PROMPT.format(
                    history=(system_prompt if self.prepend_system_prompt else [])+chat_history
                    )
                too_long, prompt_tokens = self._is_too_long(prompt)
                if not too_long: break
                chat_history = chat_history[1:]
            
            logger.debug(f'Final Prompt ({prompt_tokens} tokens):\n{prompt}')
            logger.debug('Generating...')
            
            result = self.text_generation_pipe(prompt)
            result = result[0]['generated_text']
            output_tokens = len(self.tokenizer.tokenize(result))

            logger.debug(f'Generation finished. Generated {output_tokens} tokens.')
            logger.debug(f'Result: {result}')
            
        except Exception as e:
            logger.exception('Generation failed.')
        finally:
            torch.cuda.empty_cache()
            return result