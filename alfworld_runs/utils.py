import os
import sys
import openai
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from typing import Optional, List

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import DataParallel

Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "llama"]

openai.api_key = os.getenv("OPENAI_API_KEY")

# Hugging Face setup for LLaMA with multi-GPU support
llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the actual path or model name

# Load model and tokenizer
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

# Wrap the model with DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
    llama_model = DataParallel(llama_model)
llama_model = llama_model.to("cuda")

# Create a custom generation function for multi-GPU
def llama_generate(prompt: str, max_length: int, temperature: float):
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    if isinstance(llama_model, DataParallel):
        outputs = llama_model.module.generate(  # Access the `generate` method of the underlying model
            input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=llama_tokenizer.pad_token_id,  # Ensure proper padding
            do_sample=False,
        )
    else:
        outputs = llama_model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=llama_tokenizer.pad_token_id,
            do_sample=False,
        )
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    response = openai.Completion.create(
        model="text-davinci-003",
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
def get_chat(
    prompt: str,
    model: Model,
    temperature: float = 0.0,
    max_tokens: int = 2560,
    stop_strs: Optional[List[str]] = None,
    is_batched: bool = False,
) -> str:
    if model == "llama":
        # Use the custom multi-GPU generation function
        print("Using LLaMA on multiple GPUs")
        return llama_generate(prompt, max_length=max_tokens, temperature=temperature)

    assert model != "text-davinci-003"
    messages = [
        {
            "role": "user",
            "content": prompt,
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
