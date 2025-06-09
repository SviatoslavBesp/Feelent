import random
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any, Tuple


def generate_synthetic_dataset(
        number_of_records: int,
        vector_dimension: int
) -> Dataset:
    """
    Generates a synthetic dataset for conditional text generation.

    The function creates a dataset where a custom vector dictates the style
    of the response. The vector space is divided into categories, and each
    category corresponds to a specific response style (e.g., formal, creative).

    Args:
        number_of_records (int): The total number of records to generate.
        vector_dimension (int): The dimension of the custom vector. Must be
                               divisible by the number of categories (3).

    Returns:
        datasets.Dataset: A Hugging Face Dataset object with 'prompt',
                          'response', and 'custom_vector' columns.
    """
    if vector_dimension % 3 != 0:
        raise ValueError("vector_dimension must be divisible by 3.")

    # Data sources for different styles
    source_data: Dict[str, List[Tuple[str, str]]] = {
        "science": [
            ("Что такое черная дыра?",
             "Чёрная дыра — это область пространства-времени, гравитационное притяжение которой настолько велико, что покинуть её не могут даже объекты, движущиеся со скоростью света."),
            ("Объясни фотосинтез.",
             "Фотосинтез — это сложный химический процесс преобразования энергии видимого света в энергию химических связей органических веществ при участии фотосинтетических пигментов."),
            ("Как работает GPS?",
             "Система глобального позиционирования (GPS) использует сигналы от спутников для определения точного местоположения приёмника на Земле путем трилатерации."),
            ("Что такое ДНК?",
             "Дезоксирибонуклеиновая кислота (ДНК) — это макромолекула, обеспечивающая хранение, передачу из поколения в поколение и реализацию генетической программы развития и функционирования живых организмов."),
        ],
        "history": [
            ("Расскажи о Ренессансе.",
             "Эпоха Возрождения, или Ренессанс, — это период в истории культуры Европы, пришедший на смену Средним векам и предшествующий Просвещению. Он характеризуется всплеском интереса к античному искусству и наукам."),
            ("Кто такой Юлий Цезарь?",
             "Гай Юлий Цезарь был древнеримским государственным и политическим деятелем, полководцем и писателем. Его завоевание Галлии расширило римский мир до берегов Атлантики."),
            ("Что такое Великая Китайская стена?",
             "Великая Китайская стена — это серия каменных и земляных укреплений на севере Китая, построенных для защиты китайских государств от вторжений кочевых племён."),
            ("Опиши промышленную революцию.",
             "Промышленная революция — это переход от аграрной экономики к индустриальному производству, который начался в Великобритании во второй половине XVIII века и затем распространился по всему миру."),
        ],
        "creative": [
            ("Придумай шутку про программиста.", "Почему программисты так не любят природу? Слишком много багов."),
            ("Напиши короткий стих о космосе.",
             "Средь миллиардов звёздных троп, летит бесшумно телескоп. Он ищет дом, он ищет свет, вдали от суетных планет."),
            ("Что, если бы коты умели говорить?",
             "Если бы коты умели говорить, они бы всё равно предпочитали молчать, чтобы поддерживать свой загадочный имидж. И, конечно, чтобы выпрашивать еду с ещё большей драмой."),
            ("Сочини забавный факт.",
             "Осьминоги имеют три сердца. Два качают кровь через жабры, а третье — по всему остальному телу. Видимо, поэтому они так хороши в многозадачности."),
        ]
    }

    records_list: List[Dict[str, Any]] = []
    categories: List[str] = list(source_data.keys())
    chunk_size: int = vector_dimension // 3

    for _ in range(number_of_records):
        # Choose a random category
        chosen_category: str = random.choice(categories)

        # Choose a random prompt-response pair from that category
        prompt, response = random.choice(source_data[chosen_category])

        # Generate the corresponding vector
        custom_vector = np.zeros(vector_dimension, dtype=np.float32)

        # Create "low" signal for all parts of the vector first
        for i in range(3):
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            custom_vector[start_index:end_index] = np.random.uniform(0.0, 0.2, size=end_index - start_index)

        # Create "high" signal for the chosen category's part of the vector
        category_index = categories.index(chosen_category)
        start_index = category_index * chunk_size
        end_index = start_index + chunk_size
        custom_vector[start_index:end_index] = np.random.uniform(0.8, 1.0, size=end_index - start_index)

        records_list.append({
            "prompt": prompt,
            "response": response,
            "custom_vector": custom_vector
        })

    # Convert the list of dictionaries to a Hugging Face Dataset
    final_dataset = Dataset.from_list(records_list)
    return final_dataset


if __name__ == "__main__":

    # --- Example of usage ---
    # Generate the dataset with 200 records and a vector dimension of 30
    NUMBER_OF_RECORDS = 200
    VECTOR_DIMENSION = 30

    synthetic_dataset = generate_synthetic_dataset(
        number_of_records=NUMBER_OF_RECORDS,
        vector_dimension=VECTOR_DIMENSION
    )

    # Print the dataset info and the first record to check
    print("Dataset generated successfully!")
    print(synthetic_dataset)
    print("\n--- Example Record (first entry) ---")
    # Use pandas for pretty printing of the vector
    example_record = synthetic_dataset[0]
    vector_as_series = pd.Series(example_record['custom_vector']).to_string()
    print(f"Prompt: {example_record['prompt']}")
    print(f"Response: {example_record['response']}")
    print(f"Custom Vector (len={len(example_record['custom_vector'])}):\n{vector_as_series}")

    # Check another record to see a different style
    print("\n--- Example Record (random entry) ---")
    example_record = synthetic_dataset[random.randint(0, NUMBER_OF_RECORDS - 1)]
    vector_as_series = pd.Series(example_record['custom_vector']).to_string()
    print(f"Prompt: {example_record['prompt']}")
    print(f"Response: {example_record['response']}")
    print(f"Custom Vector (len={len(example_record['custom_vector'])}):\n{vector_as_series}")