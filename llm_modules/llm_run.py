from .utils import load_config, update_args, get_token
import argparse
import os
import torch
from tqdm import tqdm

model_lookup_table = {
    'microsoft/Phi-3-mini-4k-instruct': 'hf-llm',
    'gpt-3.5-turbo': 'openai-llm'
}

def llm_ready(args):
    """
    Prepares and returns an LLM object based on the provided configuration.
    This function now supports both Hugging Face models and OpenAI Chat models.
    """
    config_file_path = args.config_path  # Path to your config file
    config_data = load_config(config_file_path)
    args = update_args(args, config_data)
    print("Updated arguments:", args)

    assert args.model_id in model_lookup_table, f"No support for model {args.model_id} added yet"

    model_type = model_lookup_table[args.model_id]
    
    # Check whether to use the OpenAI API
    if model_type == 'openai-llm':
        from .openai_support import OpenAILLM  # New OpenAI Chat wrapper
        if not args.OPENAI_APIKEY_PATH:
            raise ValueError("OpenAI API key must be provided when using OpenAI API.")
        OPENAI_APIKEY = get_token(args.OPENAI_APIKEY_PATH)
        llm = OpenAILLM(api_key=OPENAI_APIKEY, model_name=args.model_id)
        llm.load(token=OPENAI_APIKEY)  # For interface compatibility (may be a no-op for OpenAI)
        return llm
    
    elif model_type == 'hf-llm':
        from .llm_support import LLM
        os.environ["HF_HOME"] = "./.cache"
        os.environ["TORCH_HOME"] = "./.cache"

        os.chdir(args.current_dir)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA device:", torch.cuda.get_device_name(0))
        else:
            Warning("CUDA NOT AVAILABLE. RUNNING ON CPU.")
            device = "cpu"
            print("CUDA not available. Running on CPU")
        # Instantiate the Hugging Face LLM (with optional quantization)
        llm = LLM(device, args.quantization)
        HF_TOKEN = get_token(args.HF_TOKEN_PATH)
        if args.hf_checkpoint:
            llm.load_from_checkpoint(args.hf_checkpoint)
        else:
            llm.load_model_and_tokenizer(args.hf_model_path, args.model_id, HF_TOKEN)
        return llm

    else:
        raise NotImplementedError

def llm_run(model_input, args):
    """
    Runs the LLM to generate responses for the provided input.
    """
    llm = llm_ready(args)

    if args.single:
        model_input = [model_input]
    
    results = [
        llm.generate_response(conversation=conversation, max_new_tokens=args.max_new_tokens)
        for conversation in tqdm(model_input)
    ]
    
    if results is None:
        raise Exception("Failed to generate results")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General model configuration
    parser.add_argument('--language', default='english')
    # parser.add_argument('--model_id', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--model_id', default='gpt-3.5-turbo')

    parser.add_argument('--hf_model_path', default='./../../hf_models')
    parser.add_argument('--model_save_path', default='./../../hf_models/saved')
    parser.add_argument('--single', type=bool, default=False)
    parser.add_argument('--quantization', type=str, default=None)

    # Paths and configuration
    parser.add_argument('--config_path', default='./config.json')
    parser.add_argument('--cache_path', default='./../../.cache')
    parser.add_argument('--hf_checkpoint', default=None)
    parser.add_argument('--data_path', default='./../../data')
    parser.add_argument('--output_dir', default='./../../results')
    parser.add_argument('--dialect_list', nargs='+', default=None)
    parser.add_argument('--sample_count_range_start', type=int, default=0)
    parser.add_argument('--sample_count', type=int, default=None)
    parser.add_argument('--current_dir', default='./')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=200)

    # New arguments for OpenAI integration
    parser.add_argument('--use_openai_api', default=True, help="Use OpenAI Chat API instead of Hugging Face models.")

    # parser.add_argument('--openai_api_key', type=str, default=None,
    #                     help="OpenAI API key for the chat model.")
    # # Retain the HF token argument for backward compatibility
    # parser.add_argument('--HF_TOKEN_PATH', default='./path_to_hf_token.txt')

    args = parser.parse_args()

    # For demonstration purposes, define a sample conversation.
    # # Replace this with your actual conversation input as needed.
    # sample_conversation = [{"role": "user", "content": "Hello, how are you?"}]
    # responses = llm_run([sample_conversation], args)

    # print("Generated responses:")
    # for response in responses:
    #     print(response)
