import argparse
from llm_run import llm_run

def main(args):
    model_input = [
                    {"role": "system", "content": "You are a QA model. Please answer the following quesiton"},
                    {"role": "user", "content": "What is the first ever skyscraper built?"}
                ]
    results = llm_run(model_input, args)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='toxicity')
    parser.add_argument('--language', default='english')
    
    # parser.add_argument('--model_id', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--model_id', default='gpt-3.5-turbo')
    # parser.add_argument('--model_id', default='unsloth/DeepSeek-R1-GGUF')
    parser.add_argument('--hf_model_path', default='./../../hf_models')
    parser.add_argument('--model_save_path', default='./../../hf_models/saved')
    parser.add_argument('--single', type=bool, default=True)

    parser.add_argument('--quantization', type=str, default=None)
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

    parser.add_argument('--use_openai_api', default=False, help="Use OpenAI Chat API instead of Hugging Face models.")

    args = parser.parse_args()

    main(args)