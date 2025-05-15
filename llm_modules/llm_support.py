import os
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from huggingface_hub import snapshot_download
from datasets import Dataset
# import wandb
from transformers import BitsAndBytesConfig
import torch

class LLM:
    """
    A class to encapsulate Hugging Face language model functionalities,
    including loading, tokenizing, and generating responses.

    Attributes:
        model (AutoModelForCausalLM): The loaded language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        device (str): The device to use for inference ("cpu" or "cuda").
    """
    def __init__(self, device: str = "cpu", quantization: str = None):
        """
        Initialize the LLM class. 

        Args:
            device (str): The device to use for model inference. Defaults to "cpu".
        """
        self.model = None
        self.tokenizer = None
        self.device = device
        self.quantization = quantization
        print(f"LLM initiated on {self.device} with quantization={self.quantization}")

    def load_model_and_tokenizer(self, hf_model_path: str, model_id: str, token: str, redownload: bool = False):
        """
        Load or download a model and tokenizer from Hugging Face.

        Args:
            hf_model_path (str): Local directory to store the downloaded model.
            model_id (str): Hugging Face model identifier (e.g., "gpt2").
            token (str): Authentication token for private models on Hugging Face.
            redownload (bool, optional): If True, forces re-download of the model. Defaults to False.
        """
        model_path = os.path.join(hf_model_path, model_id)
        if (not os.path.exists(model_path)) or redownload:
            print(f"Downloading model from {model_id}...")
            snapshot_download(
                repo_id=model_id, 
                local_dir=model_path,
                token=token
            )

        # Set quantization config (if needed)
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load the model with quantization if specified
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config if self.quantization else None,
            trust_remote_code=True
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Model and tokenizer loaded successfully for {model_id}.")

    def load_from_checkpoint(self, hf_checkpoint: str):
        """
        Load a model and tokenizer from a specific checkpoint.

        Args:
            hf_checkpoint (str): Path to the model checkpoint.
        """
        self.model = AutoModelForCausalLM.from_pretrained(hf_checkpoint).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    def load(self, **kwargs):
        """
        Load a model and tokenizer using flexible arguments.

        Args:
            **kwargs: Keyword arguments to specify the loading method. Options:
                - hf_checkpoint (str): Path to the model checkpoint.
                - hf_model_path (str), model_id (str), token (str): Parameters to download or load a model.
                - redownload (bool, optional): Whether to force re-download of the model. Defaults to False.

        Returns:
            None: If a valid loading method is used.
            Exception: If the arguments are invalid.
        """
        redownload = kwargs.get("redownload", False)

        if "hf_checkpoint" in kwargs:
            self.load_from_checkpoint(kwargs["hf_checkpoint"])
            return None
        if "hf_model_path" in kwargs and "model_id" in kwargs and "token" in kwargs:
            self.load_model_and_tokenizer(
                kwargs["hf_model_path"], 
                kwargs["model_id"], 
                kwargs["token"], 
                redownload=redownload
            )
            return None
        raise Exception("Invalid arguments. Provide 'hf_checkpoint' or ('hf_model_path', 'model_id', 'token').")

    def generate_response(self, conversation: List[Dict], max_new_tokens: int = 100, **kwargs) -> Optional[str]:
        """
        Generate a response using the loaded model.

        Args:
            conversation (List[Dict]): Input conversation as a list of dictionaries.
                Example format:
                [
                    {"role": "user", "content": "Your question or input here"},
                    {"role": "assistant", "content": "Optional previous assistant message"}
                ]
            max_new_token (int, optional): Maximum number of new tokens to generate. Defaults to 100.
            **kwargs: Additional optional arguments (e.g., full_output).

        Returns:
            Optional[str]: Generated response string. Returns None if no valid output is generated.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generating a response.")

        gen_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 5,
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id
                    }
        
        # Use the tokenizer's apply_chat_template method
        tokenized_input = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        # Generate response
        gen_tokens = self.model.generate(
            tokenized_input,
            **gen_config
        )
        
        input_seq = self.tokenizer.decode(tokenized_input[0], skip_special_tokens=True)
        output_seq = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        if "full_output" in kwargs:
            return output_seq
        if len(input_seq) < len(output_seq):
            output_seq = output_seq[len(input_seq):]
            return output_seq.strip()
        else:
            raise Exception("No valid output generated from the model.")


class LLMFineTuner:
    """
    A class to fine-tune a pre-trained language model using the Hugging Face Transformers library
    with integrated logging to WandB.

    Attributes:
        llm (LLM): An instance of the LLM class containing the pre-trained model and tokenizer.
        output_dir (str): Directory to save the fine-tuned model.
    """
    def __init__(self, llm: LLM, output_dir: str, wandb_project: Optional[str] = None, wandb_run_name: Optional[str] = None):
        """
        Initialize the fine-tuner with an LLM object and optional WandB configuration.

        Args:
            llm (LLM): An instance of the LLM class.
            output_dir (str): Directory to save the fine-tuned model.
            wandb_project (Optional[str]): Name of the WandB project for logging. Defaults to None.
            wandb_run_name (Optional[str]): Name of the WandB run. Defaults to None.
        """
        if llm.model is None or llm.tokenizer is None:
            raise ValueError("LLM object must have a loaded model and tokenizer.")
        
        self.llm = llm
        self.output_dir = output_dir

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize WandB if project name is provided
        # if wandb_project:
        #     wandb.init(project=wandb_project, name=wandb_run_name)

    def fine_tune(
        self,
        train_data: Dataset,
        eval_data: Optional[Dataset] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        save_steps: int = 100,
        save_total_limit: int = 2,
        logging_steps: int = 50,
        evaluation_strategy: Optional[str] = "steps",
        eval_steps: Optional[int] = 100,
        warmup_steps: int = 0
    ):
        """
        Fine-tune the LLM using the specified training and evaluation datasets.

        Args:
            train_data (Dataset): The training dataset in Hugging Face `Dataset` format.
            eval_data (Optional[Dataset]): The evaluation dataset (optional).
            num_train_epochs (int): Number of training epochs. Defaults to 3.
            per_device_train_batch_size (int): Batch size for training. Defaults to 8.
            per_device_eval_batch_size (int): Batch size for evaluation. Defaults to 8.
            learning_rate (float): Learning rate for optimization. Defaults to 5e-5.
            weight_decay (float): Weight decay for regularization. Defaults to 0.01.
            save_steps (int): Save checkpoint every `save_steps`. Defaults to 100.
            save_total_limit (int): Maximum number of checkpoints to keep. Defaults to 2.
            logging_steps (int): Log training metrics every `logging_steps`. Defaults to 50.
            evaluation_strategy (Optional[str]): Evaluation strategy ("steps", "epoch", or None). Defaults to "steps".
            eval_steps (Optional[int]): Evaluate every `eval_steps` steps. Defaults to 100.
            warmup_steps (int): Number of warmup steps for learning rate scheduler. Defaults to 0.

        Returns:
            Trainer: The Hugging Face `Trainer` object used for fine-tuning.
        """
        # Data collator for sequence-to-sequence tasks
        data_collator = DataCollatorForSeq2Seq(self.llm.tokenizer, model=self.llm.model)

        # Define training arguments with WandB integration
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            warmup_steps=warmup_steps,
            # report_to="wandb",  # Enable WandB logging
            report_to=None,  # Enable WandB logging
            load_best_model_at_end=True,
            save_strategy="steps"
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.llm.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.llm.tokenizer,
            data_collator=data_collator
        )

        # Fine-tune the model
        trainer.train()

        return trainer

    def save_fine_tuned_model(self):
        """
        Save the fine-tuned model and tokenizer.

        The output directory for saving is specified during initialization.
        """
        self.llm.model.save_pretrained(self.output_dir)
        self.llm.tokenizer.save_pretrained(self.output_dir)
        print(f"Fine-tuned model and tokenizer saved to {self.output_dir}")