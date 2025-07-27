from typing import List, Optional

from pydantic import BaseModel, Field
import random

class ModelCharacter(BaseModel):
    """
    Model character description.
    """
    keywords: List[str] = Field(..., description="список ключевых слов для описания [характера модели]")
    personality: str = Field(..., description="описание личности модели")


class SystemInstructionPrototype(BaseModel):
    """
    Prototype for system instructions.
    """
    task: str = Field(..., description="описание задачи для модели")
    keywords: List[str] = Field(..., description="список ключевых слов для задачи")
    character: ModelCharacter = Field(..., description="описание характера модели, который будет использоваться в системе инструкций")


class Message(BaseModel):
    """
    Message model for dataset item generation.
    """
    role: str = Field(..., description="роль отправителя сообщения (например, 'user' или 'assistant')")
    content: str = Field(..., description="содержимое сообщения")
    event:Optional[str] = Field(None, description="событие, связанное с сообщением (например, 'start', 'end')")


class Messages(BaseModel):
    conversation_length: int = Field(..., description="длина разговора в сообщениях")
    events: List[str] = Field(..., description="список событий, связанных с сообщениями")
    topic: Optional[str] = Field(None, description="тема разговора, если применимо")
    rag_data:Optional[str] = Field(None, description="данные RAG (retrieval-augmented generation), если применимо")
    messages: Optional[List[Message]] = Field(..., description="список сообщений в разговоре")

    def generate_chat(self) -> None:
        """
        Generates a chat conversation and assigns it to the 'messages' attribute.
        The chat always starts with the 'user' role.
        Events are randomly and uniquely distributed among the messages.
        """
        # Ensure that the number of events does not exceed the conversation length
        if len(self.events) > self.conversation_length:
            raise ValueError("The number of events cannot be greater than the conversation length.")

        # Use random.sample to get unique indexes for events
        number_of_events: int = len(self.events)
        event_indexes: List[int] = random.sample(
            population=range(self.conversation_length),
            k=number_of_events
        )
        events_map: dict[int, str] = {
            index: event for index, event in zip(event_indexes, self.events)
        }

        generated_messages: List[Message] = []
        # Standard role rotation, starting with 'user'
        roles: List[str] = ['user', 'assistant']

        for i in range(self.conversation_length):
            current_role: str = roles[i % 2]
            message_content: str = f"Message {i + 1} from {current_role}"
            message_event: Optional[str] = events_map.get(i) # .get() returns None if key is not found

            generated_messages.append(
                Message(
                    role=current_role,
                    content=message_content,
                    event=message_event
                )
            )

        self.messages = generated_messages


class Emotion(BaseModel):
    name: str = Field(..., description="название эмоции (например, 'радость', 'грусть')")
    antagonist: Optional[List[str]] = Field(None, description="антагонистическая эмоция, если применимо")


class DatasetItemGenerationRequest(BaseModel):
    """
    Request model for generating a dataset item.
    """
    emotions: List[Emotion] = Field(..., description="список эмоций которые должна определить модель")
    system_instruction: SystemInstructionPrototype = Field(..., description="прототип системной инструкций для модели")
    messages:Messages = Field(..., description="сообщения для генерации набора данных")


class DatasetItemGenerationParameters(BaseModel):
    """
    Parameters for generating a dataset item.
    """
    max_conversation_length: int = Field(10, description="максимальная длина разговора в сообщениях")
    min_conversation_length: int = Field(2, description="минимальная длина разговора в сообщениях")

    available_emotions: List[Emotion] = Field(..., description="список доступных эмоций для генерации набора данных")

    conversations_topics: List[str] = Field(..., description="список тем разговоров для генерации набора данных")
    generate_rag_data: bool = Field(False, description="флаг для генерации данных RAG (retrieval-augmented generation)")

