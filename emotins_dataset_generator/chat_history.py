import random
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field


# --- User-defined Event class ---
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


# --- Enums and Core Models ---

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

    # --- RAG Configuration ---
    rag_generation_prompt: Optional[str] = Field(
        None,
        description="A prompt for an LLM to generate RAG content."
    )
    rag_placement_strategy: Optional[RAGPlacementStrategy] = Field(
        None,
        description="Strategy for placing the RAG placeholder."
    )

    # --- Event Configuration (Updated to use List[Event]) ---
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

    # --- Helper Methods ---

    def _determine_events_for_message(
            self,
            role: str,
            remaining_global_quota: Optional[int]
    ) -> List[str]:
        """
        Determines events for a single message based on probabilities, per-message limits, and a global quota.
        """
        if remaining_global_quota == 0:
            return []

        events_list: List[Event] = self.user_events if role == 'user' else self.assistant_events
        max_per_message: int = self.max_user_events_per_message if role == 'user' else self.max_assistant_events_per_message

        successful_events = [
            event.name for event in events_list if random.random() < event.probability
        ]

        # First, limit by the per-message cap
        if len(successful_events) > max_per_message:
            successful_events = random.sample(successful_events, k=max_per_message)

        # Then, limit by the remaining global quota if it's specified
        if remaining_global_quota is not None and len(successful_events) > remaining_global_quota:
            successful_events = random.sample(successful_events, k=remaining_global_quota)

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

    # --- Main Generation Method ---

    def generate_chat(self) -> None:
        """
        Generates the chat template with all configured features.
        """
        # 1. Generate base turns with probabilistic events, respecting all limits.
        total_user_events_generated = 0
        total_assistant_events_generated = 0

        turns: List[Message] = []
        for i in range(self.conversation_length):
            role = 'user' if i % 2 == 0 else 'assistant'

            remaining_quota = None
            if role == 'user' and self.max_total_user_events is not None:
                remaining_quota = self.max_total_user_events - total_user_events_generated
            elif role == 'assistant' and self.max_total_assistant_events is not None:
                remaining_quota = self.max_total_assistant_events - total_assistant_events_generated

            events = self._determine_events_for_message(role, remaining_quota)

            if role == 'user':
                total_user_events_generated += len(events)
            else:
                total_assistant_events_generated += len(events)

            content = self._generate_prompt(role, events, use_rag=False)
            turns.append(Message(role=role, events=events, content=content))

        # 2. Prepare the final message list and apply strategies
        final_messages: List[Message] = []
        base_system_prompt = self.system_prompt if self.system_prompt else ""

        # --- Apply RAG Strategy ---
        if self.rag_generation_prompt and self.rag_placement_strategy == RAGPlacementStrategy.SYSTEM_PROMPT:
            placeholder = self._generate_rag_placeholder()
            rag_context = f"\n\nUse the following context to answer: ```{placeholder}```"

            final_system_prompt = (base_system_prompt + rag_context).strip()
            final_messages.append(Message(role="system", content=final_system_prompt))

            # All assistant messages must now use RAG
            for msg in turns:
                if msg.role == 'assistant':
                    msg.content = self._generate_prompt('assistant', msg.events, use_rag=True)
            final_messages.extend(turns)

        elif self.rag_generation_prompt and self.rag_placement_strategy == RAGPlacementStrategy.SEPARATE_MESSAGE:
            if base_system_prompt:
                final_messages.append(Message(role="system", content=base_system_prompt))

            final_messages.extend(turns)

            assistant_indices = [i for i, msg in enumerate(final_messages) if msg.role == 'assistant']
            if not assistant_indices:
                raise ValueError("Cannot use 'separate_message' with no assistant messages.")

            injection_point = random.choice(assistant_indices)
            assistant_msg = final_messages[injection_point]
            assistant_msg.content = self._generate_prompt('assistant', assistant_msg.events, use_rag=True)

            tool_message = Message(role="tool", content=self._generate_rag_placeholder())
            final_messages.insert(injection_point, tool_message)

        else:
            # No RAG, or no RAG strategy specified
            if base_system_prompt:
                final_messages.append(Message(role="system", content=base_system_prompt))
            final_messages.extend(turns)

        self.messages = final_messages


# --- Example of usage (Updated to use Event objects) ---
if __name__ == '__main__':
    print("### EXAMPLE 1: SEPARATE MESSAGE RAG STRATEGY ###")
    template = Messages(
        conversation_length=6,
        topic="Modern Web Development",
        user_events=[
            Event(name="ask_for_example", probability=0.8),
            Event(name="express_confusion", probability=0.9),
            Event(name="suggest_alternative", probability=0.3)
        ],
        assistant_events=[
            Event(name="provide_code_snippet", probability=0.7),
            Event(name="summarize_concept", probability=0.6)
        ],
        max_user_events_per_message=2,
        max_assistant_events_per_message=1,
        rag_generation_prompt="Explain the core idea of server-side rendering (SSR).",
        rag_placement_strategy=RAGPlacementStrategy.SEPARATE_MESSAGE
    )

    template.generate_chat()

    if template.messages:
        for msg in template.messages:
            events_str = str(msg.events) if msg.events else "[]"
            print(f"Role: {msg.role.upper():<10} | Events: {events_str:<35} | Content: {msg.content}")

    print("\n" + "=" * 50 + "\n")

    print("### EXAMPLE 2: CUSTOM SYSTEM PROMPT WITH SYSTEM_PROMPT RAG STRATEGY ###")
    template_with_system_prompt = Messages(
        conversation_length=4,
        topic="Pythonic Code",
        system_prompt="You are a senior Python developer who gives concise and expert advice.",
        rag_generation_prompt="Provide an example of a list comprehension that is more readable than a for-loop.",
        rag_placement_strategy=RAGPlacementStrategy.SYSTEM_PROMPT,
        user_events=[Event(name="ask_for_code_review", probability=1.0)],
        assistant_events=[Event(name="give_expert_opinion", probability=1.0)]
    )

    template_with_system_prompt.generate_chat()

    if template_with_system_prompt.messages:
        for msg in template_with_system_prompt.messages:
            events_str = str(msg.events) if msg.events else "[]"
            print(f"Role: {msg.role.upper():<10} | Events: {events_str:<25} | Content: {msg.content}")

    print("\n" + "=" * 50 + "\n")

    print("### EXAMPLE 3: GLOBAL EVENT LIMITS ###")
    template_with_global_limits = Messages(
        conversation_length=10,
        topic="Data Science Careers",
        user_events=[Event(name="ask_about_salary", probability=1.0)],
        assistant_events=[Event(name="mention_job_market", probability=1.0)],
        max_total_user_events=2,
        max_total_assistant_events=3
    )

    template_with_global_limits.generate_chat()

    if template_with_global_limits.messages:
        user_event_count = 0
        assistant_event_count = 0
        for msg in template_with_global_limits.messages:
            if msg.role == 'user': user_event_count += len(msg.events)
            if msg.role == 'assistant': assistant_event_count += len(msg.events)

            events_str = str(msg.events) if msg.events else "[]"
            print(f"Role: {msg.role.upper():<10} | Events: {events_str:<25} | Content: {msg.content}")
        print("-" * 20)
        print(f"Total User Events: {user_event_count} (Limit was 2)")
        print(f"Total Assistant Events: {assistant_event_count} (Limit was 3)")

