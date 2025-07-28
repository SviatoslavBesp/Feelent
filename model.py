import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from unsloth import FastLanguageModel


class EmotionUnslothModel(nn.Module):
    """
    An unsloth-optimized wrapper that includes a trainable vector projector.
    This class takes a raw, fixed-size emotion vector, projects it to the
    model's hidden dimension, and then injects it into the forward pass.
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

        Args:
            model_name_or_path (str): The name or path of the base model.
            raw_emotion_vector_size (int): The dimension of the input emotion vector.
            lora_rank (int): The rank for LoRA decomposition.
            lora_alpha (int): The alpha parameter for LoRA.
            use_4bit (bool): Whether to load the model in 4-bit.
            max_seq_length (int): The maximum sequence length for the model.
        """
        super().__init__()
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
            cache_dir="./model_cache",
        )
        model_hidden_size = self.model.config.hidden_size
        self.vector_projector = nn.Linear(
            in_features=raw_emotion_vector_size,
            out_features=model_hidden_size,
            bias=False
        )
        self.vector_projector.to("cuda", self.model.dtype)
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
        )
        for param in self.vector_projector.parameters():
            param.requires_grad = True
        self.peft_model.print_trainable_parameters()

    def save_pretrained(
            self,
            save_directory: str
    ) -> None:
        """
        Saves the PEFT model adapters and the custom vector projector's state.

        Args:
            save_directory (str): The directory where the model components will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)

        self.peft_model.save_pretrained(save_directory)
        print(f"PEFT adapters saved to {save_directory}")

        self.tokenizer.save_pretrained(save_directory)
        print(f"Tokenizer saved to {save_directory}")

        projector_path = os.path.join(save_directory, "vector_projector.pth")
        torch.save(self.vector_projector.state_dict(), projector_path)
        print(f"Vector projector saved to {projector_path}")

    def load_pretrained(
            self,
            load_directory: str
    ) -> None:
        """
        Loads the PEFT model adapters and the custom vector projector's state.
        This method should be called on an already initialized model object.

        Args:
            load_directory (str): The directory from which to load the model components.
        """
        projector_path = os.path.join(load_directory, "vector_projector.pth")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Vector projector state file not found at {projector_path}")

        projector_state_dict = torch.load(projector_path, map_location=self.model.device)
        self.vector_projector.load_state_dict(projector_state_dict)
        self.vector_projector.to(self.model.device, dtype=self.model.dtype)
        print(f"Vector projector loaded from {projector_path}")

        self.peft_model.load_adapter(load_directory)
        print(f"PEFT adapters loaded from {load_directory}")

    def text_to_embeddings(
            self,
            text_input: str,
    ) -> np.array:
        # formatted_prompt_for_inference = prompt_template.format(inference_prompt, "")
        inputs = self.tokenizer(text_input,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.model.config.max_position_embeddings
                                ).to("cuda")

        embedding_layer = self.peft_model.get_input_embeddings()
        token_embeddings = embedding_layer(inputs.input_ids)



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
        # Project the raw emotion vector to the model's hidden dimension
        projected_vector = self.vector_projector(emotion_vector)

        # Get the standard token embeddings from the input_ids
        embedding_layer = self.peft_model.get_input_embeddings()

        # --- FIX 1: Convert input_ids to embeddings ---
        # The original code was trying to concatenate integer IDs with float embeddings.
        token_embeddings = embedding_layer(input_ids)

        # Add a sequence dimension to the projected vector to make it [batch_size, 1, hidden_size]
        projected_vector_unsqueezed = projected_vector.unsqueeze(1)

        # Concatenate the projected emotion vector at the beginning of the token embeddings
        combined_embeddings = torch.cat(
            [projected_vector_unsqueezed, token_embeddings],
            dim=1  # Concatenate along the sequence dimension
        )

        # --- FIX 2: Create an attention mask for the new token that is batch-size aware ---
        # The original code hardcoded a batch size of 1. This makes it dynamic.
        batch_size = attention_mask.shape[0]
        control_token_attention_mask = torch.ones(
            (batch_size, 1),  # Shape: [batch_size, 1] for our single new token
            dtype=torch.long,
            device=attention_mask.device
        )

        # Concatenate the new attention mask with the original one
        attention_mask = torch.cat(
            [control_token_attention_mask, attention_mask],
            dim=1
        )

        # --- FIX 3: Adjust labels if they are provided ---
        # Since we added a token at the beginning, we need to shift the labels
        # by adding a "don't care" token (-100) at the start.
        if labels is not None:
            # Create a tensor of -100s with the shape [batch_size, 1]
            prefix_labels = torch.full(
                (batch_size, 1), -100, dtype=torch.long, device=labels.device
            )
            # Concatenate with the original labels
            labels = torch.cat([prefix_labels, labels], dim=1)

        # Pass the combined inputs to the model
        model_outputs = self.peft_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return model_outputs

    def get_embeddings(
            self,
            text_input: str
    ) -> torch.Tensor:
        """
        Generates a single vector embedding for a given raw text input.
        This is done by tokenizing the text, passing it through the base model,
        and then performing masked mean pooling on the last hidden state.

        Args:
            text_input (str): The raw text to be embedded.

        Returns:
            torch.Tensor: A 1-dimensional tensor representing the text embedding.
        """
        # Set the model to evaluation mode and disable gradients for inference
        self.peft_model.eval()
        with torch.no_grad():
            # Tokenize the input text and move tensors to the correct device
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings
            ).to(self.peft_model.device)

            # Pass the tokenized input through the base model to get hidden states
            # We use peft_model here which internally calls the base model.
            outputs = self.peft_model(**inputs, output_hidden_states=True)

            # Get the last hidden state from the model output
            # Shape: [batch_size, sequence_length, hidden_size]
            last_hidden_state = outputs.last_hidden_state

            # Perform masked mean pooling to get a single vector representation
            # 1. Get the attention mask from the inputs
            attention_mask = inputs['attention_mask']

            # 2. Expand mask to match the shape of the hidden states for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            # 3. Sum the hidden states where the mask is 1
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)

            # 4. Sum the mask to get the count of actual tokens (to avoid dividing by padding)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

            # 5. Calculate the mean
            mean_pooled_embeddings = sum_embeddings / sum_mask

        # Return the model to its original mode (e.g., training) if necessary
        self.peft_model.train()

        # Squeeze to get a 1D vector [hidden_size] and return
        return mean_pooled_embeddings.squeeze()