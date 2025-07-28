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
    Manages the emotional and physical state of a character using a multiplicative update model.py.
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
    - 0.1: Легкое душевное спокойствие, мимолетная мысль о чем-то приятном.
    - 0.2: Чувство удовлетворения от хорошо выполненной задачи, легкая улыбка.
    - 0.3: Приятное тепло внутри, вызванное хорошей новостью или комплиментом.
    - 0.4: Хорошее настроение, желание делиться позитивом, смех в компании друзей.
    - 0.5: Явное чувство счастья, искренний смех, мир кажется ярче.
    - 0.6: Ощущение восторга от красивого вида или прослушивания любимой музыки.
    - 0.7: Сильная радость, почти эйфория, от большого достижения или встречи с любимым человеком.
    - 0.8: Чувство триумфа, ликование, желание прыгать от счастья.
    - 0.9: Глубокое чувство блаженства, полное погружение в момент счастья.
    - 1.0: Экстаз, всепоглощающее чувство единения со вселенной, пиковое переживание.""",

        'sadness': """Чувство утраты и разочарования.
    - 0.1: Легкая меланхолия, ностальгическое воспоминание.
    - 0.2: Небольшое разочарование, несбывшаяся мелкая надежда.
    - 0.3: Подавленное настроение, нежелание общаться.
    - 0.4: Чувство горечи, наворачивающиеся слезы.
    - 0.5: Заметная печаль, плач, ощущение тяжести в груди.
    - 0.6: Ощущение утраты, тоска по кому-то или чему-то.
    - 0.7: Глубокая скорбь, безутешные рыдания.
    - 0.8: Чувство безысходности, апатия ко всему происходящему.
    - 0.9: Душевная боль, ощущение пустоты внутри.
    - 1.0: Невыносимое страдание, отчаяние, на грани нервного срыва.""",

        'anger': """Реакция на препятствие или несправедливость.
    - 0.1: Внутреннее недовольство, несогласие с чем-то.
    - 0.2: Легкое раздражение, например, от громкого звука.
    - 0.3: Досада, сжатые кулаки, желание возразить.
    - 0.4: Злость, повышенный тон голоса, резкие движения.
    - 0.5: Явный гнев, обвинения, "кровь кипит".
    - 0.6: Желание что-то сломать или ударить.
    - 0.7: Сильная ярость, крик, агрессивные выпады.
    - 0.8: Потеря контроля, словесная агрессия, оскорбления.
    - 0.9: Неконтролируемая ярость, желание применить физическую силу.
    - 1.0: Состояние аффекта, полное помутнение рассудка, разрушительные действия.""",

        'fear': """Реакция на опасность, включает тревогу.
    - 0.1: Легкое предчувствие неладного, мурашки по коже.
    - 0.2: Настороженность, беспокойство, постоянное оглядывание.
    - 0.3: Тревога, неприятный холодок в животе.
    - 0.4: Заметный страх, учащенное сердцебиение, потливость ладоней.
    - 0.5: Испуг от резкого звука, желание спрятаться.
    - 0.6: Чувство опасности, адреналиновый всплеск.
    - 0.7: Паника, затрудненное дыхание, желание убежать.
    - 0.8: Оцепенение от страха, невозможность пошевелиться.
    - 0.9: Сильный ужас, ощущение неминуемой гибели.
    - 1.0: Абсолютный террор, потеря связи с реальностью, шок.""",

        'surprise': """Реакция на неожиданное событие.
    - 0.1: Легкое любопытство, поднятая бровь.
    - 0.2: Замешательство от неожиданной информации.
    - 0.3: Недоумение, "не может быть!".
    - 0.4: Явное удивление, открытый рот.
    - 0.5: Изумление от неожиданного подарка или события.
    - 0.6: Восклицание, всплеск руками.
    - 0.7: Ошеломление, временная потеря дара речи.
    - 0.8: Сильное потрясение от невероятного известия.
    - 0.9: Полная дезориентация, неспособность понять происходящее.
    - 1.0: Шок, как от чуда или катастрофы.""",

        'disgust': """Чувство неприязни к чему-либо.
    - 0.1: Легкая брезгливость, нежелание прикасаться к чему-то.
    - 0.2: Неприязнь к запаху или виду чего-либо.
    - 0.3: Желание отвернуться, сморщенный нос.
    - 0.4: Явное отвращение, комментарий "фу, какая гадость".
    - 0.5: Физическое желание отстраниться от источника отвращения.
    - 0.6: Чувство омерзения, тошнотворный комок в горле.
    - 0.7: Сильная тошнота, позывы к рвоте.
    - 0.8: Моральное отвращение к поступку или человеку.
    - 0.9: Неконтролируемое чувство осквернения.
    - 1.0: Физическая рвотная реакция, полное неприятие.""",

        'trust': """Ощущение безопасности и уверенности в ком-либо или чем-либо.
    - 0.1: Готовность выслушать, но с большой долей скепсиса.
    - 0.2: Осторожное согласие на небольшое сотрудничество.
    - 0.3: Предоставление минимальной личной информации.
    - 0.4: Чувство, что на человека можно положиться в простых вопросах.
    - 0.5: Базовое доверие, как к коллеге или знакомому. Готовность к открытому взаимодействию.
    - 0.6: Возможность поделиться незначительным секретом.
    - 0.7: Сильное доверие, уверенность в человеке, как в друге.
    - 0.8: Готовность доверить важное дело или ценную вещь.
    - 0.9: Полная уверенность, готовность поделиться сокровенными мыслями и чувствами.
    - 1.0: Абсолютное, безоговорочное доверие, как к самому себе.""",

        'anticipation': """Интерес и волнение по поводу будущего события.
    - 0.1: Мимолетная мысль о предстоящем событии.
    - 0.2: Слабый интерес, "посмотрим, что будет".
    - 0.3: Легкое волнение перед встречей или поездкой.
    - 0.4: Любопытство, желание поскорее узнать результат.
    - 0.5: Явное ожидание, нетерпение, частая проверка времени.
    - 0.6: Воодушевление, планирование деталей предстоящего события.
    - 0.7: Сильное предвкушение, "бабочки в животе".
    - 0.8: Почти праздничное настроение в ожидании чего-то очень хорошего.
    - 0.9: Напряженное ожидание, невозможность думать о чем-либо другом.
    - 1.0: Всепоглощающее ожидание, жизнь "на паузе" до наступления события.""",

        'emotional_arousal': """Состояние повышенной эмоциональной активности, энтузиазма, волнения, азарта или желания.
    - 0.1: Легкая заинтересованность, вовлеченность в разговор.
    - 0.2: Повышение внимания, азарт при решении сложной задачи.
    - 0.3: Энтузиазм по поводу новой идеи или проекта.
    - 0.4: Волнение перед выступлением или важным событием.
    - 0.5: Заметное воодушевление, горящие глаза, активная жестикуляция.
    - 0.6: Страстное желание чего-либо или кого-либо.
    - 0.7: Сильное эмоциональное возбуждение, экзальтация, как на концерте любимой группы.
    - 0.8: Эмоциональный подъем, ощущение "море по колено".
    - 0.9: Почти аффективное состояние, пик страсти или вдохновения.
    - 1.0: Потеря самоконтроля от переполняющих эмоций, экстатическое состояние.""",

        'energy_level': """Уровень физической и ментальной бодрости или истощения.
    - 0.1: Глаза слипаются, невозможность сфокусироваться.
    - 0.2: Сонливость, зевота, желание прилечь.
    - 0.3: Вялость, тяжесть в теле, все делается с усилием.
    - 0.4: Низкая концентрация, потребность в кофе или отдыхе.
    - 0.5: Нормальный уровень бодрости для повседневной активности.
    - 0.6: Легкость в теле, готовность к физической работе или спорту.
    - 0.7: Повышенная энергия, прилив сил, высокая продуктивность.
    - 0.8: Ощущение, что можно "свернуть горы".
    - 0.9: Перевозбуждение, сложно усидеть на месте.
    - 1.0: Гиперактивность, нервное напряжение от избытка энергии.""",

        'pain': """Общее ощущение физической боли.
    - 0.1: Легкий дискомфорт, почти незаметный (затекшая нога).
    - 0.2: Слабая, ноющая боль (небольшой синяк, легкая царапина).
    - 0.3: Раздражающая боль, которую можно игнорировать (укус комара).
    - 0.4: Умеренная боль, отвлекающая внимание (несильная головная боль).
    - 0.5: Постоянная, заметная боль, мешающая, но терпимая (боль в горле).
    - 0.6: Сильная боль, требующая обезболивающего (зубная боль).
    - 0.7: Острая боль, мешающая концентрироваться и двигаться (мигрень, растяжение).
    - 0.8: Очень сильная, пронзающая боль (перелом, ожог).
    - 0.9: Мучительная боль, вызывающая крик или слезы (почечная колика).
    - 1.0: Невыносимая, адская боль, на грани потери сознания.""",

        'muscle_tension': """Степень напряжения или расслабленности мышц.
    - 0.1: Ощущение легкой скованности после долгого сидения.
    - 0.2: Напряжение в шее или плечах от стресса.
    - 0.3: Заметное напряжение в челюсти или сжатые кулаки.
    - 0.4: Общая скованность в теле, неловкость движений.
    - 0.5: "Каменные" мышцы в спине или ногах.
    - 0.6: Напряжение, вызывающее легкую боль или дискомфорт.
    - 0.7: Сильное мышечное напряжение, дрожь в конечностях.
    - 0.8: Мышечные спазмы, сводящие мышцы.
    - 0.9: Острая боль от сильного спазма, невозможность расслабиться.
    - 1.0: Судороги, полная потеря контроля над мышцами.""",

        'trembling': """Непроизвольное дрожание тела.
    - 0.1: Едва заметный внутренний тремор, "поджилки трясутся".
    - 0.2: Легкая дрожь в руках от холода или волнения.
    - 0.3: Видимое подрагивание пальцев или губ.
    - 0.4: Дрожь в коленях.
    - 0.5: Заметное дрожание рук, которое сложно скрыть.
    - 0.6: Сильная дрожь, озноб.
    - 0.7: Стук зубов от холода или страха.
    - 0.8: Дрожь всего тела, которую невозможно контролировать.
    - 0.9: Сильные конвульсивные подергивания.
    - 1.0: Неконтролируемая, сильная дрожь, конвульсии.""",

        'hunger': """Потребность в пище или воде.
    - 0.1: Мимолетная мысль о еде.
    - 0.2: Легкое желание перекусить.
    - 0.3: Посасывание под ложечкой, начало урчания в животе.
    - 0.4: Явный голод, мысли постоянно возвращаются к еде.
    - 0.5: Громкое урчание в животе, чувство пустоты.
    - 0.6: Раздражительность от голода ("hangry").
    - 0.7: Дискомфорт, легкая слабость, сложно сконцентрироваться.
    - 0.8: Сильный, почти болезненный голод.
    - 0.9: Головокружение, слабость в ногах.
    - 1.0: Мучительный голод, на грани обморока.""",

        'physiological_arousal': """Физиологическое возбуждение: физические реакции организма на стимулы, такие как изменение пульса, дыхания, потоотделения.
    - 0.1: Легкое оживление, реакция на интересный звук.
    - 0.2: Незначительное учащение пульса при быстрой ходьбе.
    - 0.3: Учащенное дыхание после подъема по лестнице.
    - 0.4: Потливость ладоней перед важной встречей.
    - 0.5: Заметно учащенное сердцебиение от испуга или волнения.
    - 0.6: "Кровь бросилась в лицо" от смущения или гнева.
    - 0.7: Сильное сердцебиение, одышка, адреналиновый всплеск (реакция "бей или беги").
    - 0.8: Ощущение жара во всем теле, интенсивное потоотделение.
    - 0.9: Пульс стучит в висках, прерывистое дыхание.
    - 1.0: Пиковое состояние, максимальная мобилизация организма, на грани физических возможностей.""",

        'pleasure': """Ощущение физически-приятного воздействия.
    - 0.1: Приятное ощущение от теплого ветерка на коже.
    - 0.2: Удовольствие от расчесывания волос или почесывания.
    - 0.3: Наслаждение от чашки горячего чая в холодный день.
    - 0.4: Приятная усталость в мышцах после хорошей тренировки.
    - 0.5: Явное удовольствие от вкусной еды или расслабляющего массажа.
    - 0.6: Наслаждение от теплой ванны или душа.
    - 0.7: Сильное удовольствие от объятий с любимым человеком.
    - 0.8: Блаженство, нега, полное расслабление.
    - 0.9: Интенсивное, почти экстатическое наслаждение.
    - 1.0: Экстаз, оргазм, пик физического удовольствия."""
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
        # self._trust: float = 0.5 # todo training setup
        self._trust: float = self._EPSILON
        self._anticipation: float = self._EPSILON
        self._emotional_arousal: float = self._EPSILON
        # self._energy_level: float = 0.8 # todo training setup
        self._energy_level: float =self._EPSILON
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
                deviations_list.append({"name": key, "value": current_value,
                                        "description": self._ATTRIBUTE_DESCRIPTIONS.get(key,
                                                                                        "No description available.")})
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
        Updates one or more state values using a new averaging model.py.
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
        vector_list = [round(getattr(self, f"_{key}"), 2) for key in ordered_keys]
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

    def get_descriptions_from_vector(
            self,
            state_vector: NDArray[np.float32]
    ) -> List[str]:
        """
        Takes a state vector and returns a list of human-readable descriptions
        for each active state, corresponding to the closest intensity level.

        Args:
            state_vector (NDArray[np.float32]): The state vector to describe.

        Returns:
            List[str]: A list of formatted strings describing the active states.
        """
        descriptions: List[str] = []
        all_keys = self._EMOTION_KEYS + self._PHYSICAL_SENSATION_KEYS

        if len(state_vector) != len(all_keys):
            return ["Error: Input vector length does not match the number of states."]

        for i, value in enumerate(state_vector):
            # Only process states with a significant value.
            if value > 0.01:
                key = all_keys[i]
                full_description_text = self._ATTRIBUTE_DESCRIPTIONS.get(key)

                if full_description_text:
                    lines = full_description_text.split('\n')
                    # The first line is the general title, skip it.
                    description_lines = [line.strip() for line in lines[1:] if line.strip().startswith('-')]

                    best_match_line = ""
                    min_diff = float('inf')

                    for line in description_lines:
                        try:
                            # Parse the line, e.g., "- 0.7: Сильная радость..."
                            parts = line.split(':', 1)
                            level_str = parts[0].replace('-', '').strip()
                            level_float = float(level_str)

                            diff = abs(value - level_float)
                            if diff < min_diff:
                                min_diff = diff
                                # Format the output string
                                best_match_line = f"{key.capitalize()}: {line.lstrip('- ').strip()}"
                        except (ValueError, IndexError):
                            # Ignore lines that don't fit the format
                            continue

                    if best_match_line:
                        descriptions.append(best_match_line)

        return descriptions

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

    print(">>> Setting pain=0.7")
    npc_character.update_state(**{"pain": 0.7})

    print("--- Updated State ---")
    print(npc_character)

    # --- Get and print the vector ---
    current_vector = npc_character.to_vector()
    print("--- Current State Vector ---")
    print(current_vector)
    print()

    # --- Get human-readable descriptions from the vector ---
    print("--- Descriptions from Vector ---")
    readable_descriptions = npc_character.get_descriptions_from_vector(current_vector)
    for desc in readable_descriptions:
        print(desc)

