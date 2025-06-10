import random
import math
import json
import numpy as np
from numpy.typing import NDArray
from typing import Any

class EmotionalState:
    """
    Manages the emotional and physical state of a character using a multiplicative update model.
    It ensures that all related feelings are recalculated when one or more are manually set.
    """
    # A small constant to avoid division-by-zero issues.
    _EPSILON = 1e-9

    # --- Attribute Keys Definition ---
    # Defines the keys for programmatic access, e.g., for random selection or vectorization.
    # The order here is canonical and used for the to_vector() method.
    _EMOTION_KEYS: tuple[str, ...] = (
        'joy', 'sadness', 'anger', 'fear', 'surprise',
        'disgust', 'trust', 'anticipation', 'emotional_arousal'
    )
    _PHYSICAL_SENSATION_KEYS: tuple[str, ...] = (
        'energy_level', 'pain', 'muscle_tension', 'trembling',
        'hunger', 'physiological_arousal', 'pleasure'
    )

    # Data structure for attribute descriptions.
    _ATTRIBUTE_DESCRIPTIONS: dict[str, str] = {
        'joy': """Чувство удовольствия и удовлетворения.
    - 0.0: Полное отсутствие радости, апатия, ангедония.
    - 0.25: Легкое чувство удовлетворения, мимолетная улыбка.
    - 0.5: Явное чувство счастья, хорошее настроение, смех.
    - 0.75: Сильная радость, эйфория, восторг.
    - 1.0: Экстаз, всепоглощающее чувство блаженства.""",

        'sadness': """Чувство утраты и разочарования.
    - 0.0: Полное отсутствие печали.
    - 0.25: Легкая грусть, меланхолия.
    - 0.5: Заметная печаль, подавленность, могут быть слезы.
    - 0.75: Глубокая скорбь, горе.
    - 1.0: Невыносимое страдание, отчаяние, душевная боль.""",

        'anger': """Реакция на препятствие или несправедливость.
    - 0.0: Полное спокойствие, отсутствие гнева.
    - 0.25: Легкое раздражение, досада.
    - 0.5: Явный гнев, злость, повышение голоса.
    - 0.75: Сильная ярость, желание действовать агрессивно.
    - 1.0: Неконтролируемая ярость, состояние аффекта.""",

        'fear': """Реакция на опасность, включает тревогу.
    - 0.0: Полное отсутствие страха и тревоги, ощущение абсолютной безопасности.
    - 0.25: Легкое беспокойство, настороженность.
    - 0.5: Заметный страх, тревога, учащенное сердцебиение.
    - 0.75: Сильный испуг, паника.
    - 1.0: Ужас, оцепенение от страха, ощущение неминуемой гибели.""",

        'surprise': """Реакция на неожиданное событие.
    - 0.0: Полное отсутствие удивления, все предсказуемо.
    - 0.25: Легкое недоумение, любопытство.
    - 0.5: Явное удивление, поднятые брови, открытый рот.
    - 0.75: Сильное изумление, ошеломление.
    - 1.0: Шок, полная дезориентация от неожиданности.""",

        'disgust': """Чувство неприязни к чему-либо.
    - 0.0: Полное отсутствие отвращения.
    - 0.25: Легкая неприязнь, брезгливость.
    - 0.5: Явное отвращение, желание отстраниться от источника.
    - 0.75: Сильное омерзение, тошнота.
    - 1.0: Физическая рвотная реакция, неконтролируемое отвращение.""",

        'trust': """Ощущение безопасности и уверенности в ком-либо или чем-либо.
    - 0.0: Полное недоверие, подозрительность, ожидание обмана.
    - 0.25: Осторожность, наличие сомнений.
    - 0.5: Базовое доверие, готовность к открытому взаимодействию.
    - 0.75: Сильное доверие, полная уверенность.
    - 1.0: Абсолютное, безоговорочное доверие, полная уязвимость.""",

        'anticipation': """Интерес и волнение по поводу будущего события.
    - 0.0: Полное безразличие к будущему, отсутствие ожиданий.
    - 0.25: Слабый интерес, легкое любопытство.
    - 0.5: Явное ожидание, нетерпение.
    - 0.75: Сильное предвкушение, воодушевление.
    - 1.0: Напряженное, всепоглощающее ожидание, "затаив дыхание".""",

        'emotional_arousal': """Состояние повышенной эмоциональной активности, энтузиазма, волнения, азарта или желания.
    - 0.0: Полное эмоциональное спокойствие, апатия, безразличие.
    - 0.25: Легкая эмоциональная вовлеченность, интерес.
    - 0.5: Заметное воодушевление, энтузиазм, азарт.
    - 0.75: Сильное эмоциональное возбуждение, страсть, экзальтация.
    - 1.0: Состояние аффекта, пик эмоционального переживания, потеря самоконтроля.""",

        'energy_level': """Уровень физической и ментальной бодрости или истощения.
    - 0.0: Полное истощение, крайняя усталость, неспособность к действию.
    - 0.25: Сонливость, вялость, низкая концентрация.
    - 0.5: Нормальный уровень бодрости, готовность к повседневным делам.
    - 0.75: Повышенная энергия, прилив сил, высокая продуктивность.
    - 1.0: Гиперактивность, перевозбуждение, состояние "на взводе".""",

        'pain': """Общее ощущение физической боли.
    - 0.0: Полное отсутствие каких-либо болевых ощущений.
    - 0.25: Легкий дискомфорт, слабая, ноющая боль (например, легкий ушиб).
    - 0.5: Умеренная боль, которая отвлекает, но терпима (например, головная боль).
    - 0.75: Сильная боль, мешающая концентрироваться (например, зубная боль, перелом).
    - 1.0: Невыносимая, адская боль, на грани потери сознания (например, отрывание конечности, почечная колика).""",

        'muscle_tension': """Степень напряжения или расслабленности мышц.
    - 0.0: Полное мышечное расслабление, как в глубокой медитации.
    - 0.25: Легкое напряжение в отдельных группах мышц (например, в шее).
    - 0.5: Заметное общее напряжение, скованность движений.
    - 0.75: Сильное мышечное напряжение, спазмы, "каменные" мышцы.
    - 1.0: Судороги, острая боль от мышечного спазма.""",

        'trembling': """Непроизвольное дрожание тела.
    - 0.0: Полное отсутствие дрожи.
    - 0.25: Едва заметный внутренний тремор.
    - 0.5: Видимое дрожание рук или других частей тела.
    - 0.75: Сильная дрожь, озноб, стук зубов.
    - 1.0: Неконтролируемая, сильная дрожь всего тела, конвульсии.""",

        'hunger': """Потребность в пище или воде.
    - 0.0: Полная сытость, возможно, переедание.
    - 0.25: Легкое чувство голода, мысль о перекусе.
    - 0.5: Явный голод, урчание в животе.
    - 0.75: Сильный голод, дискомфорт, раздражительность.
    - 1.0: Мучительный голод, слабость, головокружение.""",

        'physiological_arousal': """Физиологическое возбуждение: физические реакции организма на стимулы, такие как изменение пульса, дыхания, потоотделения.
    - 0.0: Состояние полного покоя. Пульс и дыхание ровные и медленные.
    - 0.25: Легкое оживление. Незначительное учащение пульса и дыхания.
    - 0.5: Заметное возбуждение. Учащенное сердцебиение, поверхностное дыхание (реакция "бей или беги").
    - 0.75: Сильное возбуждение. Сильное сердцебиение, одышка, потоотделение.
    - 1.0: Пиковое состояние. Экстремально высокий пульс, максимальная мобилизация систем организма.""",

        'pleasure': """Ощущение физически-приятного воздействия.
    - 0.0: Отсутствие каких-либо приятных физических ощущений.
    - 0.25: Легкое приятное ощущение (например, теплый ветерок).
    - 0.5: Явное физическое удовольствие (например, от вкусной еды, массажа).
    - 0.75: Сильное, интенсивное наслаждение, близкое к блаженству.
    - 1.0: Экстатическое физическое удовольствие, оргазм."""
    }

    def __init__(
            self,
            relationship_map: dict[str, dict[str, float]],
            ignore_dependencies: bool = False
    ) -> None:
        """
        Initializes the character's emotional state with default values.

        Args:
            relationship_map (dict[str, dict[str, float]]):
                A map defining how emotions and sensations affect each other, where weights act as exponents.
            ignore_dependencies (bool, optional):
                If True, the system will not recalculate dependent states. Defaults to False.
        """
        self.relationship_map: dict[str, dict[str, float]] = relationship_map
        self.ignore_dependencies: bool = ignore_dependencies

        # All "zero" states are initialized with a tiny non-zero value
        # to allow them to be affected by multiplication.
        self._joy: float = self._EPSILON
        self._sadness: float = self._EPSILON
        self._anger: float = self._EPSILON
        self._fear: float = self._EPSILON
        self._surprise: float = self._EPSILON
        self._disgust: float = self._EPSILON
        self._trust: float = 0.5
        self._anticipation: float = self._EPSILON
        self._emotional_arousal: float = self._EPSILON
        self._energy_level: float = 0.8
        self._pain: float = self._EPSILON
        self._muscle_tension: float = self._EPSILON
        self._trembling: float = self._EPSILON
        self._hunger: float = self._EPSILON
        self._physiological_arousal: float = self._EPSILON
        self._pleasure: float = self._EPSILON

        # Store the initial state for future comparisons.
        self._default_state: dict[str, float] = {
            key: getattr(self, f"_{key}") for key in (self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS)
        }

    def _clamp(
            self,
            value: float
    ) -> float:
        """
        Clamps a value between 0.0 and 1.0.
        """
        return max(0.0, min(1.0, value))

    def update_state(
            self,
            **kwargs: Any
    ) -> None:
        """
        Updates one or more state values.
        If ignore_dependencies is False, it recalculates all dependent values using a multiplicative model.
        Otherwise, it only sets the specified values.
        """
        if self.ignore_dependencies:
            for key, value in kwargs.items():
                private_key = f"_{key}"
                if hasattr(self, private_key):
                    setattr(self, private_key, self._clamp(value))
            return

        # --- CORRECTED LOGIC ---
        # Populate original_values ONLY with emotional state attributes, not all internal variables.
        all_state_keys = self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS
        original_values = {f"_{key}": getattr(self, f"_{key}") for key in all_state_keys}

        multipliers = {key: 1.0 for key in original_values.keys()}

        for key, new_value in kwargs.items():
            private_key = f"_{key}"
            if private_key in original_values:
                old_value = original_values[private_key]
                safe_old_value = max(old_value, self._EPSILON)
                change_ratio = new_value / safe_old_value

                if key in self.relationship_map:
                    for target_key, weight in self.relationship_map[key].items():
                        private_target_key = f"_{target_key}"
                        if private_target_key in multipliers:
                            effect_multiplier = change_ratio ** weight
                            multipliers[private_target_key] *= effect_multiplier
            else:
                print(f"Warning: Attribute '{key}' not found in EmotionalState.")

        for private_key, original_value in original_values.items():
            new_val = original_value * multipliers[private_key]
            setattr(self, private_key, self._clamp(new_val))

        for key, value in kwargs.items():
            private_key = f"_{key}"
            if private_key in original_values:
                setattr(self, private_key, self._clamp(value))

    def to_vector(
            self
    ) -> NDArray[np.float32]:
        """
        Returns the character's current state as a NumPy array.
        """
        ordered_keys = self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS
        vector_list = [getattr(self, f"_{key}") for key in ordered_keys]
        return np.array(vector_list, dtype=np.float32)

    def get_deviations_with_details(
            self
    ) -> list[dict[str, Any]]:
        """
        Returns a list of dictionaries for all attributes whose current value
        differs from their default initialization value.
        """
        deviations_list: list[dict[str, Any]] = []
        all_keys = self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS

        for key in all_keys:
            current_value = getattr(self, f"_{key}")
            default_value = self._default_state.get(key, 0.0)

            if not math.isclose(current_value, default_value):
                deviations_list.append({
                    "name": key,
                    "value": current_value,
                    "description": self._ATTRIBUTE_DESCRIPTIONS.get(key, "No description available.")
                })

        return deviations_list

    @property
    def emotions(self) -> list[str]:
        return list(self._EMOTION_KEYS)

    @property
    def physical_sensations(self) -> list[str]:
        return list(self._PHYSICAL_SENSATION_KEYS)

    # --- Properties for read-only access to all state values ---
    @property
    def joy(self) -> float:
        return self._joy

    @property
    def sadness(self) -> float:
        return self._sadness

    @property
    def anger(self) -> float:
        return self._anger

    @property
    def fear(self) -> float:
        return self._fear

    @property
    def surprise(self) -> float:
        return self._surprise

    @property
    def disgust(self) -> float:
        return self._disgust

    @property
    def trust(self) -> float:
        return self._trust

    @property
    def anticipation(self) -> float:
        return self._anticipation

    @property
    def emotional_arousal(self) -> float:
        return self._emotional_arousal

    @property
    def energy_level(self) -> float:
        return self._energy_level

    @property
    def pain(self) -> float:
        return self._pain

    @property
    def muscle_tension(self) -> float:
        return self._muscle_tension

    @property
    def trembling(self) -> float:
        return self._trembling

    @property
    def hunger(self) -> float:
        return self._hunger

    @property
    def physiological_arousal(self) -> float:
        return self._physiological_arousal

    @property
    def pleasure(self) -> float:
        return self._pleasure

    def __str__(self) -> str:
        """Provides a string representation of the current state for easy printing."""
        core_emotions = (
            f"  Core Emotions:\n"
            f"    Joy/Счастье: {self.joy:.2f}, Sadness/Печаль: {self.sadness:.2f}, Anger/Гнев: {self.anger:.2f}\n"
            f"    Fear/Страх: {self.fear:.2f}, Surprise/Удивление: {self.surprise:.2f}, Trust/Доверие: {self.trust:.2f}\n"
            f"    Disgust/Отвращение: {self.disgust:.2f}, Anticipation/Ожидание: {self.anticipation:.2f}\n"
            f"    Emotional Arousal/Возбуждение: {self.emotional_arousal:.2f}\n"
        )
        physical_sensations = (
            f"  Physical Sensations:\n"
            f"    Energy/Энергия: {self.energy_level:.2f}, Pain/Боль: {self.pain:.2f}, Hunger/Голод: {self.hunger:.2f}\n"
            f"    Muscle Tension/Напряжение: {self.muscle_tension:.2f}, Trembling/Дрожь: {self.trembling:.2f}\n"
            f"    Physiological Arousal/Возбуждение (физ.): {self.physiological_arousal:.2f}, Pleasure/Удовольствие: {self.pleasure:.2f}"
        )
        return f"===== Character Emotional State =====\n{core_emotions}{physical_sensations}\n"


# --- Example Usage ---
if __name__ == "__main__":

    DEFAULT_RELATIONSHIP_MAP: dict[str, dict[str, float]] = {
        "joy": {"sadness": -1.5, "energy_level": 0.5, "trust": 0.2, "pleasure": 1.2},
        "fear": {"physiological_arousal": 1.2, "trembling": 1.5, "trust": -1.2, "sadness": 0.3},
        "pain": {"sadness": 1.2, "energy_level": -1.0, "anger": 0.4, "pleasure": -2.0},
    }

    npc_character = EmotionalState(relationship_map=DEFAULT_RELATIONSHIP_MAP, ignore_dependencies=True)
    print("--- Initial State ---")
    print(npc_character)

    print(">>> Setting fear=0.8, joy=0.5")
    npc_character.update_state(fear=0.8, joy=0.5)
    print("--- Updated State ---")
    print(npc_character)

    print("--- Deviations from Default State (Detailed) ---")
    active_states = npc_character.get_deviations_with_details()

    # Pretty-print the JSON-like structure for clear output
    print(json.dumps(active_states, indent=2, ensure_ascii=False, default=lambda x: float(f"{x:.4f}")))
