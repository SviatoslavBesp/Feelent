
import unsloth
from unsloth import FastLanguageModel

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from typing import Optional, List, Dict, Any

from transformers import AutoConfig, TrainingArguments
from trl import SFTTrainer

# ==============================================================================
# 1. DEFINITION OF THE CUSTOM MODEL WRAPPER
# ==============================================================================
class EmotionUnslothModel(nn.Module):
    """
    An unsloth-optimized wrapper that includes a trainable vector projector.
    This class takes a raw, fixed-size emotion vector, projects it to the
    model.py's hidden dimension, and then injects it into the forward pass.
    """
    def __init__(
        self,
        model_name_or_path: str,
        raw_emotion_vector_size: int,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        use_4bit: bool = True,
        max_seq_length: int = 2048,
    ):
        """
        Initializes the EmotionUnslothModel with a vector projector.
        """
        super().__init__()

        # Load unsloth model.py and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
            cache_dir="./model_cache",
        )
        model_hidden_size = self.model.config.hidden_size

        # Define the vector projector
        self.vector_projector = nn.Linear(
            in_features=raw_emotion_vector_size,
            out_features=model_hidden_size,
            bias=False
        )
        self.vector_projector.to("cuda",self.model.dtype)

        # Apply LoRA using unsloth's function
        self.peft_model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            # modules_to_save=["vector_projector"], # <--- УДАЛИТЕ ЭТУ СТРОКУ
        )

        # --- ДОБАВЬТЕ ЭТОТ БЛОК ---
        # Manually unfreeze the projector weights after creating the PEFT model.py.
        # This makes them trainable without using the restricted 'modules_to_save'.
        for param in self.vector_projector.parameters():
            param.requires_grad = True
        # ---------------------------

        self.peft_model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        emotion_vector: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass with projection and injection.
        """
        projected_vector = self.vector_projector(emotion_vector)
        embedding_layer = self.peft_model.get_input_embeddings()
        token_embeddings = embedding_layer(input_ids)
        combined_embeddings = token_embeddings + projected_vector.unsqueeze(1)

        model_outputs = self.peft_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return model_outputs

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        emotion_vector: torch.Tensor,
        **generation_kwargs: Any
    ) -> List[int]:
        """
        Generates text conditioned on a prompt and an emotion vector.
        """
        if "pad_token_id" not in generation_kwargs:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        return self.peft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emotion_vector=emotion_vector,
            **generation_kwargs
        )

# ==============================================================================
# 2. DEFINITION OF THE CUSTOM DATA COLLATOR
# ==============================================================================

from transformers import DataCollatorForLanguageModeling

class DataCollatorForEmotionLM(DataCollatorForLanguageModeling):
    """
    Custom data collator that handles tokenizing text and stacking emotion vectors.
    """
    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes a list of features to create a batch.
        """
        emotion_vectors = [feature.pop("emotion_vector") for feature in features]
        batch = super().__call__(features)
        batch['emotion_vector'] = torch.stack(emotion_vectors)
        return batch

# ==============================================================================
# 3. MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    # --- Configuration ---
    MODEL_NAME = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    RAW_EMOTION_VECTOR_SIZE = 12

    # --- Model Initialization ---
    print("Initializing the model.py...")
    emotion_model_wrapper = EmotionUnslothModel(
        model_name_or_path=MODEL_NAME,
        raw_emotion_vector_size=RAW_EMOTION_VECTOR_SIZE,
        max_seq_length=MAX_SEQ_LENGTH
    )
    model_dtype = emotion_model_wrapper.model.dtype

    # --- Data Preparation ---
    print("Preparing the dataset...")
    # For Llama-3 instruct model.py, we should use its specific chat template
    prompt_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

    raw_data = [
        {"text": prompt_template.format("Write a happy poem about spring.", "The sunbeams dance, a joyful sight,\nNew flowers bloom in colors bright."), "emotion": "happy"},
        {"text": prompt_template.format("Describe a spooky, abandoned mansion.", "The old manor stood in chilling dread,\nWhere silent ghosts and shadows tread."), "emotion": "spooky"},
        {"text": prompt_template.format("Compose a short, joyful song about a river.", "The river flows, a happy tune,\nBeneath the sunny afternoon."), "emotion": "happy"},
        {"text": prompt_template.format("Tell a short, eerie tale about a forest at night.", "Deep in the woods, when moonlight fails,\nA whisper rides on chilling gales."), "emotion": "spooky"},
    ]
    dataset = Dataset.from_list(raw_data)

    # Create and map emotion vectors to the dataset
    emotion_mapping = {
        "happy": torch.from_numpy(np.random.rand(RAW_EMOTION_VECTOR_SIZE) * 0.1),
        "spooky": torch.from_numpy(np.random.rand(RAW_EMOTION_VECTOR_SIZE) * -0.1)
    }

    # CORRECTED: Define the function inside main() to access model_dtype
    def add_emotion_vector(example: Dict[str, Any]) -> Dict[str, Any]:
        """Adds the emotion vector to a dataset example with the correct dtype on CPU."""
        em_vector = emotion_mapping[example["emotion"]]
        # The tensor should be created with the model.py's dtype, but remain on the CPU.
        example["emotion_vector"] = em_vector.to(dtype=model_dtype)
        return example

    # CORRECTED: Call .map() with only the function argument
    dataset = dataset.map(add_emotion_vector)

    # --- Pre-processing and Tokenization ---
    def preprocess_function(example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes the text and prepares labels."""
        tokenized_example = emotion_model_wrapper.tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors=None,
        )
        tokenized_example["labels"] = tokenized_example["input_ids"][:]
        return tokenized_example

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # --- Trainer Setup ---
    print("Setting up the trainer...")
    data_collator = DataCollatorForEmotionLM(
        tokenizer=emotion_model_wrapper.tokenizer,
        mlm=False
    )

    trainer = SFTTrainer(
        model=emotion_model_wrapper.peft_model,
        tokenizer=emotion_model_wrapper.tokenizer,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=50,  # Increase for real training
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
        ),
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished!")

      # --- Inference Example ---

    print("\n--- Пример генерации текста (инференс) ---")

    inference_prompt = "Tell me about a sunny day."
    inference_emotion = "happy"


    formatted_prompt = prompt_template.format(inference_prompt, "")
    inputs = emotion_model_wrapper.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    # Manually prepare `inputs_embeds` for inference
    # 1. Get the emotion vector and project it
    emotion_vector_tensor = emotion_mapping[inference_emotion].unsqueeze(0).to("cuda",dtype=model_dtype)
    projected_vector = emotion_model_wrapper.vector_projector(emotion_vector_tensor)

    # 2. Get the token embeddings from the input_ids
    embedding_layer = emotion_model_wrapper.peft_model.get_input_embeddings()
    token_embeddings = embedding_layer(inputs.input_ids)

    # 3. Combine them to create final inputs_embeds
    combined_embeddings = token_embeddings + projected_vector.unsqueeze(1)

    # Call generate with `inputs_embeds` instead of `input_ids`.
    generated_ids = emotion_model_wrapper.peft_model.generate(
        inputs_embeds=combined_embeddings,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        use_cache=True,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=emotion_model_wrapper.tokenizer.eos_token_id,
    )

    generated_text = emotion_model_wrapper.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n--- Сгенерированный текст ---")
    print(generated_text)



if __name__ == "__main__":
    # Run the main function to start the training and inference process
    main()