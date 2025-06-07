import asyncio
from dataclasses import asdict
import json
from typing import Dict, List, Union
from openai import AsyncOpenAI

# Предполагается, что SensationVectorGenerator находится в этом файле
# и его конструктор будет обновлен для приема новых параметров.
from senstaion_vector import SensationVector, SensationVectorGenerator


class SituationGenerator:
    """
    Генератор описаний ситуаций на основе эмоциональных и физических ощущений.
    """

    def __init__(
            self,
            openai_client: AsyncOpenAI,
            model: str = "gpt-4o-mini"
    ):
        self.client = openai_client
        self.model = model
        # Системный промпт остается без изменений, он превосходен
        self.system_prompt = """
        You are a psychological profiler and master screenwriter. Your task is to transform a set of raw sensory and emotional data points into a deeply realistic and logically coherent micro-scene.

        **CRITICAL RULES:**

        1.  **SHOW, DON'T TELL:** The `narrative` must demonstrate the character's state through actions, metaphors, and sensory details. DO NOT directly explain their feelings (e.g., instead of "he felt unfocused," write "his gaze drifted past her shoulder to the window").
        2.  **NO DATA REFERENCES:** **NEVER** mention the numerical values (`0.58`, `-0.34`) or the parameter names (`Confidence`, `Energy Level`) in any of the output fields intended for prose (`summary`, `narrative`, `monologue`). These values are for your internal analysis only.

        **1. Glossary of Intensity (Mandatory for use)**

        You MUST use these definitions to interpret the numerical values. The hypothesized background and the final narrative must be consistent with this scale.

        * **Pain (0.0 to 1.0):** `0.0-0.2`: Insignificant; `0.2-0.5`: Distracting; `0.5-0.8`: Strong; `0.8-1.0`: Unbearable.
        * **Energy Level (0.0 to 1.0):** `0.0-0.3`: Apathy/Exhaustion; `0.7-1.0`: Hyperactivity/Restlessness.
        * **Confidence (-1.0 to 1.0):** `-1.0 to -0.5`: Acute insecurity; `-0.5 to 0.0`: Mild self-doubt.
        * **Openness (-1.0 to 1.0):** `-1.0 to -0.5`: Completely defensive/distrustful; `-0.5 to 0.0`: Reserved/cautious.

        **2. Processing Steps**

        1.  **Analyze Data & Contradiction:** Analyze all input data through the lens of the Glossary. Identify and state the Core Contradiction.
        2.  **Distill Keywords:** Based on your analysis, generate two lists of 3-5 keywords or short phrases each:
            * **Emotional Keywords:** Should be evocative and specific (e.g., 'Guarded vulnerability', 'Anxious energy').
            * **Physical Keywords:** Should be concrete and descriptive (e.g., 'Pulsing hand pain', 'Restless legs').
        3.  **Hypothesize a Causal Background:** Construct a brief, plausible backstory that occurred *recently*. This backstory must explain the Core Contradiction and be consistent with the keywords.
        4.  **Summarize Sensations:**
            * **Psychological Summary:** Elaborate on the `emotional_keywords` in a descriptive paragraph.
            * **Physical Summary:** Elaborate on the `physical_keywords` in a descriptive paragraph.
        5.  **Write the Scene:**
            * **Narrative:** Write the scene, adhering strictly to the "Critical Rules". The narrative must be a direct consequence of the background hypothesis and embody the keywords and summaries.
            * **Internal Monologue:** Write a short, italicized internal thought that reveals the character's core conflict.

        **3. Output Format (JSON):**
        {
          "theme": "str",
          "analysis": {
            "core_contradiction": "str",
            "background_hypothesis": "str",
            "emotional_keywords": ["str", "str", ...],
            "physical_keywords": ["str", "str", ...],
            "psychological_summary": "str",
            "physical_summary": "str"
          },
          "scene": {
            "narrative": "str",
            "internal_monologue": "str"
          }
        }
        Пиши примеры на русском языке
        """

    def _build_prompt(
            self,
            theme: str,
            sensations: SensationVector,
    ) -> str:
        """
        Формирует промпт для модели
        """
        # Этот метод можно упростить, так как SensationVector, вероятно, имеет __str__ или __repr__
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
            sensations: SensationVector,
            n_generations: int = 5
    ) -> List[Dict]:
        """
        Генерирует n описаний ситуаций по заданной теме и ощущениям
        """
        prompt = self._build_prompt(
            theme=theme,
            sensations=sensations,
        )
        tasks = [self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=1,
            max_tokens=2000,
        ) for _ in range(n_generations)]
        raw_responses = await asyncio.gather(*tasks)
        raw_responses = [response.choices[0].message.content for response in raw_responses if response.choices]
        # Добавим проверку на пустой ответ
        results = [json.loads(raw_response) for raw_response in raw_responses if raw_response]
        return results

    async def generate_examples_by_topic(
            self,
            topic: str,
            n_samples: int = 5,
            llm_generations_per_sample: int = 5,
            # --- НОВЫЕ И ОБНОВЛЕННЫЕ ПАРАМЕТРЫ ---
            min_active_body_parts: int = 1,
            max_active_body_parts: int = 3,
            min_sensations_per_part: int = 1,
            max_sensations_per_part: int = 3,
            # -----------------------------------------
            seed=123,
            output_file="situations_examples.json",
            rewrite_file=False,
            batch_size: int = 10
    ):
        """Generate and optionally save situation examples for a given topic."""

        # --- ИНИЦИАЛИЗАЦИЯ ГЕНЕРАТОРА С НОВЫМИ ПАРАМЕТРАМИ ---
        generator = SensationVectorGenerator(
            number_of_samples=n_samples,
            min_active_body_parts=min_active_body_parts,
            max_active_body_parts=max_active_body_parts,
            min_sensations_per_part=min_sensations_per_part,
            max_sensations_per_part=max_sensations_per_part,
            seed=seed,
        )
        sensations: List[SensationVector] = generator.generate()

        # Run generation in parallel
        tasks = [self.generate(topic, sensation, n_generations=llm_generations_per_sample) for sensation in sensations]
        generated = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            # Await the completion of the current batch
            batch_results = await asyncio.gather(*batch)
            generated.extend(batch_results)

        examples = [
            {"theme": topic, "sensations": s.to_dict(), "results": r}
            for s, r in zip(sensations, generated)
        ]

        if output_file:
            try:
                if rewrite_file:
                    existing: List = []
                else:
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
            except FileNotFoundError:
                existing = []

            existing.extend(examples)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

        return examples


if __name__ == "__main__":
    import os

    # Рекомендуется использовать переменные окружения для ключей API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    client = AsyncOpenAI(api_key=api_key)

    situations_gen = SituationGenerator(
        openai_client=client,
        model="gpt-4o-mini"  # gpt-4.1-nano может не существовать, заменил на gpt-4o-mini
    )

    # --- ПРИМЕР ВЫЗОВА С НОВЫМИ ПАРАМЕТРАМИ ---
    asyncio.run(
        situations_gen.generate_examples_by_topic(
            topic="physical injuries",
            n_samples=10,
            llm_generations_per_sample=5,
            min_active_body_parts=1,
            max_active_body_parts=4,
            min_sensations_per_part=1,
            max_sensations_per_part=3,
            seed=3,
            output_file="injuries.json",
            rewrite_file=False,
        )
    )
