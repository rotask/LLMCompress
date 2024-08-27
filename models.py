import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralForCausalLM, BitsAndBytesConfig, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig

def get_model_and_tokenizer(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if model_name == "Mixtral":
        model = MixtralForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1", 
            quantization_config=bnb_config, 
            # load_in_4bit=True, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    elif model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2",
            quantization_config=bnb_config,
            # attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
    elif model_name == "Yi":
        model = AutoModelForCausalLM.from_pretrained(
            "01-ai/Yi-34B", 
            quantization_config=bnb_config, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B")
    
    elif model_name == "Nemo":
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-Nemo-Base-2407",
            quantization_config=bnb_config, 
            # attn_implementation="flash_attention_2", 
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Base-2407")
        
    elif model_name == "llama_2":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config, 
        attn_implementation="flash_attention_2", 
        device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")        
                        
    elif model_name == "llama_3":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
        quantization_config=bnb_config, 
        attn_implementation="flash_attention_2", 
        device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    elif model_name == "llama_3.1":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B",
        quantization_config=bnb_config, 
        attn_implementation="flash_attention_2", 
        device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
                    
    elif model_name == "Mistral_7B":
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
            quantization_config=bnb_config, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        
        # Add more models as needed
        
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model, tokenizer