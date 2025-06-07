import asyncio
from typing import Dict, List, Union
from openai import AsyncOpenAI

from senstaion_vector import SensationVector, SensationVectorGenerator


class SituationGenerator:
    """
    Генератор описаний ситуаций на основе эмоциональных и физических ощущений.
    """

    def __init__(
            self,
            openai_client: AsyncOpenAI,
            model: str = "gpt-4o"
    ):
        self.client = openai_client
        self.model = model
        self.system_prompt = """
        You are an AI tasked with generating highly realistic character situation descriptions based on internal sensation data.
        
        Your input is:
        1. A short **theme** (e.g., “romantic date”, “combat”, “stage performance”)
        2. A detailed list of **emotional** and **physical** sensations.
        
        Your goal is to create an emotionally immersive and physically grounded narrative that reflects these values **as literally and specifically as possible**.
        
        --- Interpretation rules ---
        All values range from -1.0 to 1.0:
        - Emotional states (Valence, Arousal, etc.):
          - -1.0 = extreme negative (despair, terror)
          -  0.0 = neutral
          -  1.0 = extreme positive (euphoria, full engagement)
        
        - Physical states:
          - Pain: -1.0 = no pain, 1.0 = unbearable pain
          - Temperature: -1.0 = freezing, 1.0 = burning
          - Numbness: 1.0 = completely numb
          - Pressure, Tension: 1.0 = maximum tightness or weight
        
        --- Task ---
        1. Interpret each value explicitly — explain its meaning and how it affects the character.
        2. For each body part, explain how the physical sensation feels and how it affects behavior.
        3. For emotions, explain how they manifest in thoughts or behavior.
        4. Then, construct a short but vivid situation based on the combined interpretation.
        
        
        In the "situation" field:
        - Describe a scene in which these emotional and physical states were *naturally* caused.
        - Include what just happened or is happening (what action, interaction, or event).
        - Reflect how these sensations affect the character's behavior.
        - Use specific sensory language: sounds, touch, smell, movement.
        - Show consequences: what might happen next, what the character is preparing for or reacting to.
        
        Avoid:
        - Generalized feeling-only descriptions.
        - Repeating the same sentences from the summaries.
        - Generic phrases like “he felt alive” or “he was in the moment”.
        
        --- Output format ---
        Return a structured JSON object:
        {
          "theme": str,
          "situation": str,         //  up to one sentence
          "emotions_summary": str, 
          "physical_summary": str,  // map each body part and sensation to a physical feeling
        } 
        """

    def _build_prompt(
            self,
            theme: str,
            sensations: Dict[str, Union[float, Dict]]
    ) -> str:
        """
        Формирует промпт для модели
        """
        prompt = f"""
        Generate a realistic and immersive description of a situation where a character feels the following sensations. Focus on emotions, physical reactions, inferred context, and environmental factors.
        
        Theme: {theme}
        ______________
        {sensations}
        """
        return prompt

    async def generate(
            self,
            theme: str,
            sensations: Dict[str, Union[float, Dict]],
            n: int = 1
    ) -> List[Dict]:
        """
        Генерирует n описаний ситуаций по заданной теме и ощущениям
        """
        prompt = self._build_prompt(theme, sensations)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=800,
            n=n,
        )

        results = []
        for choice in response.choices:
            try:
                result = choice.message.function_call.arguments if hasattr(choice.message,
                                                                           "function_call") else choice.message.content
                if isinstance(result, str):
                    result = eval(result)  # если модель вернула как stringified dict
                results.append(result)
            except Exception as e:
                print(f"Ошибка парсинга результата: {e}")
        return results

    async def generate_examples_by_topic(
            self,
            topic: str,
            n_samples: int = 5,
            max_active_body_parts: int = 3,
            sensation_activation_chance=0.3,
            seed=123,
            output_file="situations_examples.json",
            rewrite_file=False
    ):
        pass
