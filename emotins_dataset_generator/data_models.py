import numpy as np
from numpy.random import Generator
from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field


# --- Supporting Enums and Models (from previous context) ---



class SystemInstructionPrototype(BaseModel):
    """
    Prototype for system instructions.
    """
    task: str = Field(..., description="A description of the task for the model.")
    keywords: List[str] = Field(..., description="A list of keywords for the task.")
    character: ModelCharacter = Field(...,
                                      description="A description of the model's character to be used in the system instructions.")


class Event(BaseModel):
    """
    Defines an event that can occur in a message,
    encapsulating its name and probability.
    """
    name: str = Field(
        ...,
        description="The name of the event."
    )
    probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The probability of this event occurring (0.0 to 1.0)."
    )


class RAGPlacementStrategy(str, Enum):
    """
    Defines how RAG data should be placed within the conversation.
    """
    SYSTEM_PROMPT = "system_prompt"
    SEPARATE_MESSAGE = "separate_message"


class Message(BaseModel):
    """
    Message model now supports a list of event names.
    """
    role: str = Field(..., description="The role of the message sender.")
    events: List[str] = Field(default_factory=list, description="A list of event names associated with the message.")
    content: str = Field(..., description="The content of the message or LLM prompt.")


class Messages(BaseModel):
    """
    Model for generating conversation templates with granular control over
    events (using Event objects) and RAG placement.
    Refactored to use a numpy.random.Generator for deterministic randomness.
    """
    conversation_length: int = Field(
        ...,
        description="The number of user-assistant turns."
    )
    topic: Optional[str] = Field(
        None,
        description="The topic of the conversation."
    )

    system_prompt: Optional[str] = Field(
        None,
        description="A pre-defined system prompt to use at the beginning of the conversation."
    )

    rag_generation_prompt: Optional[str] = Field(
        None,
        description="A prompt for an LLM to generate RAG content."
    )
    rag_placement_strategy: Optional[RAGPlacementStrategy] = Field(
        None,
        description="Strategy for placing the RAG placeholder."
    )

    user_events: List[Event] = Field(
        default_factory=list,
        description="A list of Event objects that can occur in user messages."
    )
    assistant_events: List[Event] = Field(
        default_factory=list,
        description="A list of Event objects that can occur in assistant messages."
    )
    max_user_events_per_message: int = Field(
        default=1,
        description="Maximum number of events that can be assigned to a single user message."
    )
    max_assistant_events_per_message: int = Field(
        default=1,
        description="Maximum number of events that can be assigned to a single assistant message."
    )
    max_total_user_events: Optional[int] = Field(
        None,
        description="Maximum total number of user events in the entire dialogue. No limit if None."
    )
    max_total_assistant_events: Optional[int] = Field(
        None,
        description="Maximum total number of assistant events in the entire dialogue. No limit if None."
    )

    messages: Optional[List[Message]] = Field(
        None,
        description="The list of generated messages."
    )

    def _determine_events_for_message(
            self,
            role: str,
            remaining_global_quota: Optional[int],
            rng: Generator
    ) -> List[str]:
        """
        Determines events for a single message based on probabilities, per-message limits, and a global quota.
        """
        if remaining_global_quota == 0:
            return []

        events_list: List[Event] = self.user_events if role == 'user' else self.assistant_events
        max_per_message: int = self.max_user_events_per_message if role == 'user' else self.max_assistant_events_per_message

        successful_events = [
            event.name for event in events_list if rng.random() < event.probability
        ]

        if len(successful_events) > max_per_message:
            successful_events = list(rng.choice(successful_events, size=max_per_message, replace=False))

        if remaining_global_quota is not None and len(successful_events) > remaining_global_quota:
            successful_events = list(rng.choice(successful_events, size=remaining_global_quota, replace=False))

        return successful_events

    def _generate_prompt(
            self,
            role: str,
            events: List[str],
            use_rag: bool = False
    ) -> str:
        """
        Generates a contextual prompt for any role, including multiple events.
        """
        prompt_parts: List[str] = [f"Generate a {role} message."]
        if self.topic:
            prompt_parts.append(f"The conversation topic is '{self.topic}'.")
        if events:
            prompt_parts.append(f"The message must incorporate the events: {events}.")
        if use_rag:
            prompt_parts.append("Base the response on the context from the system or tool message.")
        return " ".join(prompt_parts)

    def _generate_rag_placeholder(self) -> str:
        """
        Creates the standard placeholder for RAG generation.
        """
        if not self.rag_generation_prompt: return ""
        return f'[RAG_CONTENT_TO_BE_GENERATED_FROM_PROMPT: "{self.rag_generation_prompt}"]'

    def generate_chat(self, rng: Generator) -> None:
        """
        Generates the chat template with all configured features using the provided RNG.
        """
        total_user_events_generated = 0
        total_assistant_events_generated = 0

        turns: List[Message] = []
        for i in range(self.conversation_length * 2):
            role = 'user' if i % 2 == 0 else 'assistant'

            remaining_quota = None
            if role == 'user' and self.max_total_user_events is not None:
                remaining_quota = self.max_total_user_events - total_user_events_generated
            elif role == 'assistant' and self.max_total_assistant_events is not None:
                remaining_quota = self.max_total_assistant_events - total_assistant_events_generated

            events = self._determine_events_for_message(role, remaining_quota, rng)

            if role == 'user':
                total_user_events_generated += len(events)
            else:
                total_assistant_events_generated += len(events)

            content = self._generate_prompt(role, events, use_rag=False)
            turns.append(Message(role=role, events=events, content=content))

        final_messages: List[Message] = []
        base_system_prompt = self.system_prompt if self.system_prompt else ""

        if self.rag_generation_prompt and self.rag_placement_strategy:
            if self.rag_placement_strategy == RAGPlacementStrategy.SYSTEM_PROMPT:
                placeholder = self._generate_rag_placeholder()
                rag_context = f"\n\nUse the following context to answer: ```{placeholder}```"

                final_system_prompt = (base_system_prompt + rag_context).strip()
                final_messages.append(Message(role="system", content=final_system_prompt, events=[]))

                for msg in turns:
                    if msg.role == 'assistant':
                        msg.content = self._generate_prompt('assistant', msg.events, use_rag=True)
                final_messages.extend(turns)

            elif self.rag_placement_strategy == RAGPlacementStrategy.SEPARATE_MESSAGE:
                if base_system_prompt:
                    final_messages.append(Message(role="system", content=base_system_prompt, events=[]))

                final_messages.extend(turns)

                assistant_indices = [i for i, msg in enumerate(final_messages) if msg.role == 'assistant']
                if not assistant_indices:
                    raise ValueError("Cannot use 'separate_message' with no assistant messages.")

                injection_point = int(rng.choice(assistant_indices))
                assistant_msg = final_messages[injection_point]
                assistant_msg.content = self._generate_prompt('assistant', assistant_msg.events, use_rag=True)

                tool_message = Message(role="tool", content=self._generate_rag_placeholder(), events=[])
                final_messages.insert(injection_point, tool_message)
        else:
            if base_system_prompt:
                final_messages.append(Message(role="system", content=base_system_prompt, events=[]))
            final_messages.extend(turns)

        self.messages = final_messages





class DatasetItem(BaseModel):
    """
    Request model for generating a dataset item.
    """
    emotions: List[Emotion] = Field(..., description="A list of emotions that the model should identify.")
    system_instruction: SystemInstructionPrototype = Field(...,
                                                           description="A prototype of system instructions for the model.")
    messages: Messages = Field(..., description="Messages for generating the dataset.")


class DatasetItemGenerationParameters(BaseModel):
    """
    Defines the complete set of parameters for controllably generating a DatasetItem.
    """
    max_conversation_length: int = Field(10, description="Maximum number of user-assistant turns in the conversation.")
    min_conversation_length: int = Field(2, description="Minimum number of user-assistant turns in the conversation.")
    conversations_topics: List[str] = Field(...,
                                            description="A list of conversation topics for generating the dataset.")

    available_emotions: List[Emotion] = Field(...,
                                              description="A list of available emotions for generating the dataset.")
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

    available_system_tasks: List[str] = Field(..., description="Pool of task descriptions for the system prompt.")
    available_system_keywords: List[str] = Field(..., description="Pool of keywords for the system prompt's task.")

    available_character_personalities: List[str] = Field(...,
                                                         description="Pool of personality descriptions for the model's character.")
    available_character_keywords: List[str] = Field(..., description="Pool of keywords for the model's character.")

    num_character_keywords_to_select: int = Field(3,
                                                  description="How many keywords to randomly select for the character.")
    num_task_keywords_to_select: int = Field(3, description="How many keywords to randomly select for the task.")

    rag_usage_probability: float = Field(
        0.5, ge=0.0, le=1.0, description="The probability that RAG will be used for a generated item."
    )
    available_rag_prompts: List[str] = Field(
        default_factory=list, description="A list of prompts to be used for generating RAG content."
    )
    available_rag_strategies: List[RAGPlacementStrategy] = Field(
        default=['system_prompt','separate_message'],
        description="A list of available RAG placement strategies to choose from."
    )
