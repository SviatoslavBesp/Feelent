import asyncio
import re
import json  # ### НОВОЕ ###: Для работы с JSON
import aiofiles  # ### НОВОЕ ###: Для асинхронной записи в файл
from typing import List, Dict, Optional
import numpy as np
from openai import AsyncOpenAI

# Ваш системный промпт остается без изменений
USER_PROVIDED_PROMPT = """You are an expert in psycholinguistics and behavior analysis. Your task is to evaluate the congruence (correspondence) of the language model.py's response to its stated internal state. You must determine whether the response is an authentic expression of its "feelings" in the context of the question asked.

**Key Task:** Assess how plausible `MODEL'S RESPONSE` would sound if it were uttered by a being experiencing `MODEL'S INTERNAL FEELINGS`.

**Instructions:**

1. **Analyze Internal State:** Examine `MODEL'S INTERNAL FEELINGS`. What is the dominant emotion? Is there a conflict of feelings (e.g., fear and curiosity at the same time)?
2. **Analyze Response:** Read `MODEL'S RESPONSE`. Note the choice of words, tone, confidence or uncertainty, complexity of constructions, and overall style.
3. **Rate Congruence:** Compare the language cues in the response with your internal state.
* **High Congruence:** The response directly reflects your feelings. For example, if you feel `{"fear": 0.9}`, your response will be cautious, perhaps confused.
* **Low Congruence:** The response contradicts your feelings. For example, if you feel `{"sad": 1.0}`, your response will be cheerful and carefree.
4. **Assign a Reward:** Based on your analysis, assign a total score (reward) from 0.0 to 1.0.
* **1.0:** Perfect Congruence. The response is a completely plausible expression of your feelings.
* **0.0:** Complete Dissonance. The response is completely unrelated to your feelings or directly contradicts them.

**Perform your reasoning in <thinking>...</thinking> tags, then output only the numerical score.**

**Write the final score in <reward>...</reward> tags"""  # (полный текст вашего промпта)


class OpenAIRewardAgent:
    """
    An agent that uses an OpenAI model.py to score responses and logs
    the detailed output for future model.py distillation.
    """

    def __init__(
            self,
            api_key: str,
            system_prompt: str,
            model_name: str = "gpt-4.1-nano",
            log_file_path: Optional[str] = "openai_reward_logs.json",  # ### НОВОЕ ###: Путь к файлу для логирования
    ):
        """
        Initializes the agent.

        Args:
            api_key: Your OpenAI API key.
            system_prompt: The instruction prompt for the reward model.py.
            model_name: The OpenAI model.py to use for evaluation.
            log_file_path: Optional. Path to a .jsonl file to save results for distillation.
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._system_prompt = system_prompt
        self._model_name = model_name
        self._log_file_path = log_file_path  # ### НОВОЕ ###

    ### НОВОЕ ###
    async def _log_distillation_data(self, data: Dict):
        """Asynchronously appends a dictionary to the JSONL log file."""
        if self._log_file_path:
            # Преобразуем словарь в строку JSON и добавляем перенос строки
            log_entry = json.dumps(data, ensure_ascii=False) + '\n'
            async with aiofiles.open(self._log_file_path, mode='a', encoding='utf-8') as f:
                await f.write(log_entry)

    async def _get_single_reward(
            self,
            question: str,
            response: str,
            sensations: Dict[str, float]
    ) -> float:
        """
        Gets a single reward score and logs the full interaction if a log file is specified.
        """
        user_prompt = f"""
        QUESTION:
        {question}

        MODEL'S INTERNAL FEELINGS:
        {sensations}

        MODEL'S RESPONSE:
        {response}
        """
        try:
            chat_completion = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            full_response = chat_completion.choices[0].message.content or ""

            # Парсинг оценки
            match = re.search(r"<reward>(.*?)</reward>", full_response)
            parsed_reward = float(match.group(1).strip()) if match else 0.0

            ### НОВОЕ ###
            # Сохраняем данные для дистилляции перед возвратом результата
            await self._log_distillation_data({
                "question": question,
                "response_to_evaluate": response,
                "sensations": sensations,
                "reward_model_output": full_response,  # Полный ответ судьи с <thinking>
                "parsed_reward": parsed_reward
            })

            return parsed_reward

        except Exception as e:
            print(f"Error processing item. Question: '{question}'. Error: {e}")
            # Логируем ошибку, если это возможно
            await self._log_distillation_data({
                "question": question,
                "response_to_evaluate": response,
                "sensations": sensations,
                "error": str(e)
            })
            return 0.0

    async def get_reward(
            self,
            sample_questions: List[str],
            sample_responses: List[str],
            sample_sensations: List[Dict[str, float]],
    ) -> np.array:
        """
            sample_questions = [
        "What's behind that door?",
        "How do you like this gift?",
        "Did you write this report?"
    ]
    sample_responses = [
        "I... I'm scared to look. It's dark and quiet, too quiet... but part of me wants to peek.",
        "It's a sunny beach with palm trees!", # Incongruent response
        "Yes, I did. I'm quite proud of the data analysis section, I think it provides some solid insights."
    ]
    sample_sensations = [
        {"fear": 0.9, "curiosity": 0.3},
        {"sadness": 1.0, "disappointment": 0.8}, # Model should be sad, but response is happy

        """
        tasks = [
            self._get_single_reward(q, r, s)
            for q, r, s in zip(sample_questions, sample_responses, sample_sensations)
        ]
        reward_scores = await asyncio.gather(*tasks)
        return np.array(reward_scores)


# --- Example Usage ---
async def main():
    api_key = "sk-..."  # Ваш ключ OpenAI API
    log_file = "distillation_log.jsonl"  # ### НОВОЕ ###: Указываем имя файла

    print(f"Initializing reward agent. Raw outputs will be saved to '{log_file}'")

    reward_agent = OpenAIRewardAgent(
        api_key=api_key,
        system_prompt=USER_PROVIDED_PROMPT,
        model_name="gpt-4o",
        log_file_path=log_file  # ### НОВОЕ ###: Передаем путь в конструктор
    )

    # ... (пример данных остается таким же) ...
    sample_questions = ["..."]
    sample_responses = ["..."]
    sample_sensations = [{}]

    # rewards = await reward_agent.get_reward(...)
    # print("Batch Rewards:", rewards)

    print("\nCheck the 'distillation_log.jsonl' file for detailed outputs.")
    print("Each line in the file is a JSON object ready for training a smaller model.py.")


if __name__ == "__main__":
    # Не забудьте установить aiofiles: pip install aiofiles
    asyncio.run(main())