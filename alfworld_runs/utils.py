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

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist

Model = Literal["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "llama"]

openai.api_key = os.getenv('OPENAI_API_KEY')

device_map = "balanced"  # Automatically balance the model across GPUs
model_name = "meta-llama/Llama-3.2-1B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)

local_rank = int(os.getenv("LOCAL_RANK", 0))  # Get the local rank from environment variables
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

llama_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct"
    # Remove the device argument to allow accelerate to handle it
)

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
        # Initialize the distributed process group (for multi-GPU usage)
        dist.init_process_group("nccl")

        # Get global rank to uniquely identify each process
        global_rank = int(os.getenv("RANK", 0))

        # Use pipeline for text generation on the specific GPU
        result = llama_pipeline(
            prompt,
            max_length=max_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,  # Ensure proper padding
            do_sample=False
        )

        # Gather responses from all GPUs
        world_size = dist.get_world_size()
        responses = [None] * world_size
        dist.all_gather_object(responses, result[0]["generated_text"])

        # Rank 0 aggregates and returns the results
        if global_rank == 0:
            aggregated_response = " ".join(responses)
            return aggregated_response

        # Clean up the distributed process group
        dist.destroy_process_group()

    # If not using LLaMA, fallback to OpenAI models
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
