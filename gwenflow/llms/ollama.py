# from typing import Optional, Union, List, Dict, Any
# import openai

from gwenflow.llms.openai import ChatOpenAI


class ChatOllama(ChatOpenAI):
 
    base_url: str
