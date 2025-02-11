import openai
from typing import List, Dict, Optional

class OpenAILLM:
    """
    A class to encapsulate the OpenAI Chat model functionality.
    This class implements an interface similar to the LLM class, enabling
    seamless integration with your existing pipeline.
    """
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAILLM with the required API key and model name.

        Args:
            api_key (str): Your OpenAI API key.
            model_name (str): The OpenAI chat model name (e.g., "gpt-3.5-turbo").
        """
        self.model_name = model_name
        self.model = None
        print(f"OpenAI initiated")

    def load(self, **kwargs):
        """
        For OpenAI models, there is no need to load a local model.
        This method exists to preserve the same interface.
        """
        assert "token" in kwargs, "Token/openai API key must be provided to access GPT models"
        self.model = openai.OpenAI(
                api_key=kwargs["token"],  # This is the default and can be omitted
            )

    def generate_response(self, conversation: List[Dict], max_new_tokens: int = 100, **kwargs) -> Optional[str]:
        """
        Generate a response using OpenAI’s ChatCompletion API.

        Args:
            conversation (List[Dict]): Input conversation formatted as a list of dictionaries.
                For example:
                    [
                        {"role": "user", "content": "Your question or input here"},
                        {"role": "assistant", "content": "Optional previous assistant message"}
                    ]
            max_new_tokens (int): Maximum number of new tokens (mapped to OpenAI's max_tokens).
            **kwargs: Additional parameters (e.g., temperature, top_p) that may override defaults.

        Returns:
            Optional[str]: Generated response text.
        """
        print(conversation)
        chat_completion = self.model.chat.completions.create(
            messages=conversation,
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content.strip()
        # # Set parameters, mapping from your original configuration.
        # temperature = kwargs.get("temperature", 0.7)
        # top_p = kwargs.get("top_p", 0.1)
        # # Note: OpenAI’s API uses 'max_tokens' instead of Hugging Face's 'max_new_tokens'
        # max_tokens = max_new_tokens

        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=conversation,
        #     temperature=temperature,
        #     top_p=top_p,
        #     max_tokens=max_tokens,
        #     # If desired, you can add additional parameters here.
        # )
        
        # # Extract and return the assistant's response.
        # return response["choices"][0]["message"]["content"].strip()
