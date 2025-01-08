import os
import sys
import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
from typing import Optional, List

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from transformers import pipeline  # For Hugging Face LLaMA

Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "llama"]

openai.api_key = os.getenv('OPENAI_API_KEY')

# Hugging Face pipeline setup for LLaMA with GPU support
llama_pipeline = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-1B-Instruct",  # Replace <path_to_llama_model> with the model's path or name
    device_map='balanced'  # Use GPU (device 0)
)

# for l, d in llama_pipeline.model.hf_device_map.items():
#     #print(llama_pipeline.model.hf_device_map)
#     #llama_pipeline.model.hf_device_map[l] = 2
#     #print(l, llama_pipeline.model.hf_device_map[l])

# nnodes = 2
# gpupnode = 2
# d = 0
# for l in llama_pipeline.model.hf_device_map:
#     llama_pipeline.model.hf_device_map[l] = d % gpupnode


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 2560, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    if model == "llama":
        # Use Hugging Face pipeline for LLaMA
        print('using llama')
        result = llama_pipeline(
            prompt,
            max_length=max_tokens,
            temperature=temperature,
            pad_token_id=llama_pipeline.tokenizer.pad_token_id,  # Ensure proper padding
            do_sample=False
        )
        return result[0]["generated_text"]
    
    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )
    return response.choices[0]["message"]["content"]