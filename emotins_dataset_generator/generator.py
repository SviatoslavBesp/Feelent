from emotins_dataset_generator.data_models import *
from emotins_dataset_generator.emotions_pack import EMOTIONS
# --- The Generator ---

class DatasetItemGenerationParameters(BaseModel):
    """
    Defines the complete set of parameters for controllably generating a DatasetItem.
    """
    # --- Conversation Structure ---
    max_conversation_length: int = Field(10, description="Maximum number of user-assistant turns in the conversation.")
    min_conversation_length: int = Field(2, description="Minimum number of user-assistant turns in the conversation.")
    conversations_topics: List[str] = Field(...,
                                            description="A list of conversation topics for generating the dataset.")

    # --- Emotion / Event Configuration ---
    available_emotions: List[Emotion] = Field(...,
                                              description="The complete pool of available emotions to draw from for generation.")
    max_emotions_in_conversation: int = Field(3,
                                              description="Maximum number of unique emotions to select for a single conversation.")

    min_event_probability: float = Field(0.1, ge=0.0, le=1.0,
                                         description="Minimum probability for an emotion event to occur.")
    max_event_probability: float = Field(0.8, ge=0.0, le=1.0,
                                         description="Maximum probability for an emotion event to occur.")

    max_user_events_per_message: int = Field(2,
                                             description="Maximum number of emotion events allowed in a single user message.")
    max_assistant_events_per_message: int = Field(2,
                                                  description="Maximum number of emotion events allowed in a single assistant message.")

    max_total_user_events: int = Field(5,
                                       description="Maximum total number of user emotion events in the entire dialogue.")
    max_total_assistant_events: int = Field(5,
                                            description="Maximum total number of assistant emotion events in the entire dialogue.")

    # --- System Instruction Generation ---
    available_system_tasks: List[str] = Field(..., description="Pool of task descriptions for the system prompt.")
    available_system_keywords: List[str] = Field(..., description="Pool of keywords for the system prompt's task.")

    available_character_personalities: List[str] = Field(...,
                                                         description="Pool of personality descriptions for the model's character.")
    available_character_keywords: List[str] = Field(..., description="Pool of keywords for the model's character.")

    num_character_keywords_to_select: int = Field(3,
                                                  description="How many keywords to randomly select for the character.")
    num_task_keywords_to_select: int = Field(3, description="How many keywords to randomly select for the task.")

    # --- RAG (Retrieval-Augmented Generation) Configuration ---
    rag_usage_probability: float = Field(
        0.5, ge=0.0, le=1.0, description="The probability that RAG will be used for a generated item."
    )
    available_rag_prompts: List[str] = Field(
        default_factory=list, description="A list of prompts to be used for generating RAG content."
    )
    available_rag_strategies: List[RAGPlacementStrategy] = Field(
        default=[RAGPlacementStrategy.SYSTEM_PROMPT, RAGPlacementStrategy.SEPARATE_MESSAGE],
        description="A list of available RAG placement strategies to choose from."
    )


# --- The Generator ---

class DatasetGenerator:
    """
    Generates DatasetItem instances based on a set of defined parameters.
    Uses a numpy.random.Generator for all stochastic operations to ensure reproducibility.
    """
    params: DatasetItemGenerationParameters
    rng: Generator

    def __init__(
            self,
            parameters: DatasetItemGenerationParameters,
            seed: Optional[int] = None
    ):
        """
        Initializes the generator with a set of parameters and an optional seed.
        """
        self.params = parameters
        self.rng = np.random.default_rng(seed)

    def _generate_system_instruction(self) -> SystemInstructionPrototype:
        """
        Creates a randomized SystemInstructionPrototype from the available pools.
        """
        char_keywords = self.rng.choice(
            self.params.available_character_keywords,
            size=min(len(self.params.available_character_keywords), self.params.num_character_keywords_to_select),
            replace=False
        ).tolist()

        task_keywords = self.rng.choice(
            self.params.available_system_keywords,
            size=min(len(self.params.available_system_keywords), self.params.num_task_keywords_to_select),
            replace=False
        ).tolist()

        character = ModelCharacter(
            keywords=char_keywords,
            personality=self.rng.choice(self.params.available_character_personalities)
        )

        instruction = SystemInstructionPrototype(
            task=self.rng.choice(self.params.available_system_tasks),
            keywords=task_keywords,
            character=character
        )
        return instruction

    def _generate_events(self, active_emotions: List[Emotion]) -> List[Event]:
        """
        Creates a list of Event objects from a given list of active emotions with randomized probabilities.
        """
        events = []
        for emotion in active_emotions:
            probability = self.rng.uniform(self.params.min_event_probability, self.params.max_event_probability)
            events.append(Event(name=emotion.name, probability=probability))
        return events

    def generate_item(self) -> DatasetItem:
        """
        Generates a single, fully-formed DatasetItem.
        """
        # 1. Generate System Instruction
        system_instruction = self._generate_system_instruction()

        # 2. Select emotions for this specific conversation from the available pool
        num_emotions_to_select = self.rng.integers(
            1,
            min(len(self.params.available_emotions), self.params.max_emotions_in_conversation) + 1
        )
        active_emotions = self.rng.choice(
            self.params.available_emotions,
            size=num_emotions_to_select,
            replace=False
        ).tolist()

        # 3. Generate Event objects based on the selected emotions
        events = self._generate_events(active_emotions)

        # 4. Basic Conversation Parameters
        conversation_length = self.rng.integers(
            self.params.min_conversation_length,
            self.params.max_conversation_length + 1
        )
        topic = self.rng.choice(self.params.conversations_topics)

        # 5. RAG Configuration
        use_rag = self.rng.random() < self.params.rag_usage_probability
        rag_prompt = None
        rag_strategy = None
        if use_rag and self.params.available_rag_prompts:
            rag_prompt = self.rng.choice(self.params.available_rag_prompts)
            rag_strategy = self.rng.choice(self.params.available_rag_strategies)

        # 6. Create the Messages object
        messages_template = Messages(
            conversation_length=conversation_length,
            topic=topic,
            system_prompt=f"Task: {system_instruction.task}. Be {system_instruction.character.personality}.",
            user_events=events,
            assistant_events=events,
            max_user_events_per_message=self.params.max_user_events_per_message,
            max_assistant_events_per_message=self.params.max_assistant_events_per_message,
            max_total_user_events=self.params.max_total_user_events,
            max_total_assistant_events=self.params.max_total_assistant_events,
            rag_generation_prompt=rag_prompt,
            rag_placement_strategy=rag_strategy
        )

        # 7. Generate the actual chat messages
        messages_template.generate_chat(rng=self.rng)

        # 8. Assemble and return the final DatasetItem
        dataset_item = DatasetItem(
            emotions=active_emotions,  # Now contains only the emotions active in this chat
            system_instruction=system_instruction,
            messages=messages_template
        )

        return dataset_item

class CharacterTrait(BaseModel):
    name: str
    antagonists: List[str] = Field(default_factory=list, description="List of emotions that are antagonistic to this trait.")


class Character(BaseModel):
    """
    Represents a character in the conversation with personality and keywords.
    """
    name: str
    personality: str = Field(..., description="Personality description of the character.")



# Example usage (can be removed or commented out)
if __name__ == '__main__':
    # Define some example parameters
    gen_params = DatasetItemGenerationParameters(
        conversations_topics=["booking a vacation", "discussing a movie", "planning a dinner party"],
        available_emotions=[
           em for em in EMOTIONS.values()
        ],
        max_emotions_in_conversation=3,  # New parameter
        available_system_tasks=["Act as a helpful assistant.", "You are a chatbot for a travel agency."],
        available_system_keywords=["friendly", "efficient", "creative", "formal"],
        available_character_personalities=["a cheerful and bubbly guide", "a calm and professional expert"],
        available_character_keywords=["empathetic", "witty", "patient", "direct"],
        num_character_keywords_to_select=2,
        num_task_keywords_to_select=2,
        rag_usage_probability=0.7,
        available_rag_prompts=["Find the best flight deals for a trip to Paris.", "Summarize the plot of 'Inception'."]
    )

    # Create a generator instance with a fixed seed for reproducibility
    generator = DatasetGenerator(parameters=gen_params, seed=42)

    # Generate a single dataset item
    new_item = generator.generate_item()

    # Print the generated item in a readable format
    print(f"Number of active emotions in this item: {len(new_item.emotions)}")
    print([e.name for e in new_item.emotions])
    print("-" * 20)
    print(new_item.model_dump_json(indent=2))