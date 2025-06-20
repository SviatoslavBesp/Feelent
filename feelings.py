import random
import math
import json
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Dict, Tuple

DEFAULT_RELATIONSHIP_MAP: dict[str, dict[str, float]] = {
    "joy": {"sadness": -1.5, "energy_level": 0.5, "trust": 0.2, "pleasure": 1.2},
    "fear": {"physiological_arousal": 1.2, "trembling": 1.5, "trust": -1.2, "sadness": 0.3},
    "pain": {"sadness": 1.03, "energy_level": -1.0, "anger": 0.4, "pleasure": -2.0},
}

STOIC_REALIST_MAP: dict[str, dict[str, float]] = {
    # --- Emotional Triggers ---
    "joy": {
        "sadness": -1.2,  # Joy directly counteracts sadness.
        "energy_level": 0.4,  # Feeling happy provides a moderate energy boost.
        "trust": 0.3,  # Happiness makes one slightly more trusting.
        "pleasure": 0.8,  # Joy is closely linked to feelings of pleasure.
        "emotional_arousal": 0.5,  # Joy creates moderate positive excitement.
    },
    "sadness": {
        "joy": -1.5,  # Sadness strongly suppresses joy.
        "energy_level": -0.8,  # Sadness is draining and reduces energy.
        "trust": -0.4,  # It's harder to trust others when feeling down.
        "anticipation": -0.5,  # Sadness dampens any sense of anticipation.
        "muscle_tension": 0.3,  # Can cause slight physical tension or heaviness.
    },
    "anger": {
        "trust": -1.2,  # Anger severely damages trust.
        "joy": -1.0,  # It's hard to feel joy when angry.
        "physiological_arousal": 0.8,  # Anger readies the body for a confrontation.
        "muscle_tension": 0.9,  # Anger leads to significant muscle tension.
        "pain": 0.2,  # High anger can manifest as a form of discomfort.
    },
    "fear": {
        "trust": -1.5,  # Fear makes one highly suspicious and distrustful.
        "joy": -1.0,  # Fear eclipses happiness.
        "physiological_arousal": 1.2,  # The core of the "fight or flight" response.
        "trembling": 0.9,  # A classic physical manifestation of fear.
        "energy_level": -0.5,  # Fear can be paralyzing and draining.
        "surprise": 0.4,  # Fear is often triggered by a surprise.
    },
    "surprise": {
        "fear": 0.3,  # A surprise can be startling and cause a bit of fear.
        "anticipation": 0.5,  # A neutral surprise raises curiosity for what's next.
        "emotional_arousal": 0.6,  # An immediate spike in emotional awareness.
    },
    "disgust": {
        "pleasure": -1.2,  # Disgust is the opposite of pleasure.
        "hunger": -1.0,  # Feeling disgusted can eliminate appetite.
        "joy": -0.6,  # Hard to be happy when disgusted.
    },
    "trust": {
        "joy": 0.4,  # Trusting someone feels good.
        "fear": -0.8,  # Feeling of safety from trust reduces fear.
        "muscle_tension": -0.5,  # Trust allows one to relax physically.
    },
    "anticipation": {
        "joy": 0.6,  # Positive anticipation is a form of joy.
        "emotional_arousal": 0.7,  # Eagerly awaiting something is exciting.
        "energy_level": 0.3,  # Gives a slight boost of energy.
    },

    # --- Physical Triggers ---
    "pain": {
        "sadness": 1.0,  # Pain is a direct cause of sadness and distress.
        "anger": 0.4,  # Can cause frustration and anger.
        "pleasure": -2.0,  # Pain is the direct opposite of physical pleasure.
        "energy_level": -1.2,  # Pain is extremely draining.
        "muscle_tension": 0.8,  # The body tenses up in response to pain.
    },
    "pleasure": {
        "joy": 1.2,  # Physical pleasure is a strong source of joy.
        "sadness": -0.8,  # It's hard to be sad when experiencing pleasure.
        "pain": -1.0,  # Pleasure and pain are mutually exclusive.
        "muscle_tension": -0.7,  # Pleasure often involves physical relaxation.
    },
    "hunger": {
        "anger": 0.3,  # Being "hangry" is a real phenomenon.
        "energy_level": -0.5,  # Lack of food leads to low energy.
        "anticipation": 0.6,  # Hunger creates anticipation for the next meal.
    },
    "energy_level": {
        "joy": 0.5,  # Having energy makes it easier to feel happy.
        "sadness": -0.5,  # High energy combats feelings of sadness.
    }
}

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
        self._ALL_STATE_KEYS: Tuple[str, ...] = self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS


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


    def get_state_deviations(self) -> List[Dict[str, Any]]:
        deviations_list: List[Dict[str, Any]] = []
        for key in self._ALL_STATE_KEYS:
            current_value = getattr(self, f"_{key}")
            default_value = self._default_state.get(key, 0.0)
            if not math.isclose(current_value, default_value, abs_tol=self._EPSILON):
                deviations_list.append({"name": key, "value": current_value, "description": self._ATTRIBUTE_DESCRIPTIONS.get(key, "No description available.")})
        return deviations_list

    def get_current_state_as_dict(self) -> Dict[str, float]:
        return {key: getattr(self, f"_{key}") for key in self._ALL_STATE_KEYS}

    def set_current_state_as_default(self) -> None:
        self._default_state = self.get_current_state_as_dict()


    def update_state(
            self,
            **kwargs: Any
    ) -> None:
        """
        Updates one or more state values using a new averaging model.
        1. States specified in kwargs are set directly (manual override).
        2. For all other states, it calculates the average influence from all active triggers.
        3. The new value is the average of the old value and the calculated influence.
        """
        # --- Handle the simple case first ---
        if self.ignore_dependencies:
            for key, value in kwargs.items():
                private_key = f"_{key}"
                if hasattr(self, private_key):
                    setattr(self, private_key, self._clamp(value))
            return

        # --- New Averaging Logic ---

        # Step 1: Store the original state before any calculations.
        original_state = self.get_current_state_as_dict()

        # Step 2: Calculate all influences from the trigger states (kwargs).
        # We store them in a dictionary where each key maps to a list of influences.
        influences: Dict[str, List[float]] = {key: [] for key in self._ALL_STATE_KEYS}

        for trigger_key, trigger_value in kwargs.items():
            # Check if this trigger has any defined relationships.
            if trigger_key in self.relationship_map:
                # Iterate through all target states affected by this trigger.
                for target_key, coefficient in self.relationship_map[trigger_key].items():
                    # Calculate the influence this trigger exerts on the target.
                    influence_value = trigger_value * coefficient
                    influences[target_key].append(influence_value)

        # Step 3: Iterate through ALL states to calculate their new values.
        for key in self._ALL_STATE_KEYS:
            # Check for manual override. If the key was in kwargs, its value is set directly.
            if key in kwargs:
                new_value = kwargs[key]
            else:
                # If not manually overridden, apply the calculation logic.
                target_influences = influences[key]

                # If there are any influences on this state...
                if target_influences:
                    # Calculate the average of all influences.
                    average_influence = sum(target_influences) / len(target_influences)

                    # Get the state's original value.
                    old_value = original_state[key]
                    if old_value <= 0.1:
                        old_value = average_influence

                    # Apply the new formula: (old_value + average_influence) / 2
                    new_value = (old_value + average_influence) / 2
                else:
                    # If there were no influences, the value remains unchanged.
                    new_value = original_state[key]

            # Set the final, clamped value.
            setattr(self, f"_{key}", self._clamp(new_value))

    def to_vector(
            self
    ) -> NDArray[np.float32]:
        """
        Returns the character's current state as a NumPy array.
        """
        ordered_keys = self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS
        vector_list = [round(getattr(self, f"_{key}"),2) for key in ordered_keys]
        return np.array(vector_list, dtype=np.float32)

    def to_dicts_list(
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

            # if not math.isclose(current_value, default_value):
            if not current_value == default_value:
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



    npc_character = EmotionalState(relationship_map=STOIC_REALIST_MAP, ignore_dependencies=False)
    print("--- Initial State ---")
    print(npc_character)

    print(">>> Setting fear=0.8, joy=0.5")
    npc_character.update_state(**{"pain":0.7})

    print("--- Updated State ---")
    print(npc_character)

    print("--- Deviations from Default State (Detailed) ---")
    active_states = npc_character.to_dicts_list()

    # Pretty-print the JSON-like structure for clear output
    print(json.dumps(active_states, indent=2, ensure_ascii=False, default=lambda x: float(f"{x:.4f}")))
