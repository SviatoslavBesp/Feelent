import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional


# --- Existing classes from your example ---
# (They remain unchanged)

class BodyPartLocalization(Enum):
    """
    Standardized names for body parts for sensation localization.
    """
    HEAD = "Head"
    FACE = "Face"
    NECK = "Neck"
    CHEST = "Chest"
    ABDOMEN = "Abdomen"
    BACK = "Back"
    LEFT_ARM = "Left Arm"
    RIGHT_ARM = "Right Arm"
    LEFT_HAND = "Left Hand"
    RIGHT_HAND = "Right Hand"
    LEFT_LEG = "Left Leg"
    RIGHT_LEG = "Right Leg"
    FEET = "Feet"


@dataclass
class BodyPartSensation:
    """
    Describes the physical sensations localized to a specific body part.
    """
    pain: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0
    tension: float = 0.0
    numbness: float = 0.0
    vibration: float = 0.0

    def vector(self) -> np.ndarray:
        """
        Convert the sensation data to a dictionary vector.
        """
        return np.array(list(asdict(self).values()), dtype=np.float32)

    def __str__(self):
        """
        String representation of the body part sensation.
        """
        answer = ""
        for key, value in asdict(self).items():
            if value != 0.0:
                answer += f"{key.capitalize()}: {round(value, 2)}\n"
        return answer.strip()


@dataclass
class SensationVector:
    """
    Represents the character's state, including detailed physical sensations.
    """
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0
    focus: float = 0.0
    openness: float = 0.0
    dominance: float = 0.0
    energy_level: float = 1.0
    physical_state: Dict[BodyPartLocalization, BodyPartSensation] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        """
        Convert the sensation vector to a dictionary vector.
        """
        base_vector = np.array([
            self.valence, self.arousal, self.confidence, self.focus,
            self.openness, self.dominance, self.energy_level
        ], dtype=np.float32)

        if not self.physical_state:
            return base_vector

        body_part_vectors = np.concatenate(
            [sensation.vector() for sensation in self.physical_state.values()]
        )
        return np.concatenate((base_vector, body_part_vectors))

    def __str__(self):
        """
        String representation of the sensation vector.
        """
        sensation = f"Valence: {self.valence:.2f}, Arousal: {self.arousal:.2f}, Confidence: {self.confidence:.2f}, Focus: {self.focus:.2f}, Openness: {self.openness:.2f}, Dominance: {self.dominance:.2f}, Energy Level: {self.energy_level:.2f}"
        physical_sensations = "Physical Sensations:"
        if not self.physical_state:
            physical_sensations += " None"
        else:
            for body_part, p_sensation in self.physical_state.items():
                sensation_str = str(p_sensation)
                if sensation_str:
                    physical_sensations += f"\n  {body_part.value}:\n    {sensation_str.replace(chr(10), chr(10) + '    ')}"

        return sensation + "\n" + physical_sensations

    def to_dict(self) -> Dict:
        """
        Convert the sensation vector to a dictionary representation.
        """
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "confidence": self.confidence,
            "focus": self.focus,
            "openness": self.openness,
            "dominance": self.dominance,
            "energy_level": self.energy_level,
            "physical_state": {
                part.value: asdict(sensation) for part, sensation in self.physical_state.items()
            }
        }

# --- Updated Generator Class ---

class SensationVectorGenerator:
    """
    Generates a list of SensationVector objects with realistic correlations
    and a distribution centered around neutral values.
    """

    def __init__(
            self,
            number_of_samples: int,
            max_active_body_parts: int = 3,
            sensation_activation_chance: float = 0.3,
            seed: Optional[int] = None
    ):
        """
        Initializes the generator.
        :param number_of_samples: The number of SensationVector objects to generate.
        :param max_active_body_parts: The maximum number of body parts that can have
                                      sensations in a single vector.
        :param sensation_activation_chance: The probability (0.0 to 1.0) for each
                                            individual sensation (like pain, temp)
                                            to be active on a body part.
        :param seed: An optional random seed for reproducibility.
        """
        self.number_of_samples = number_of_samples
        self.max_active_body_parts = max_active_body_parts
        self.sensation_activation_chance = sensation_activation_chance
        self.rng = np.random.default_rng(seed)

    def _generate_physical_state(self) -> Dict[BodyPartLocalization, BodyPartSensation]:
        """
        Generates a dictionary of sparse physical sensations for random body parts.
        """
        physical_state: Dict[BodyPartLocalization, BodyPartSensation] = {}

        # Determine how many body parts will be active, up to the specified maximum.
        # The upper bound for rng.integers is exclusive, so we add 1.
        number_of_active_parts = self.rng.integers(0, self.max_active_body_parts + 1)

        if number_of_active_parts == 0:
            return {}

        all_body_parts = list(BodyPartLocalization)
        selected_parts = self.rng.choice(all_body_parts, size=number_of_active_parts, replace=False)

        for part in selected_parts:
            sensation = BodyPartSensation()

            if self.rng.random() < self.sensation_activation_chance:
                sensation.numbness = self.rng.triangular(left=0.0, mode=0.0, right=1.0)

            max_other_sensation = 1.0 - (sensation.numbness * 0.9)

            if self.rng.random() < self.sensation_activation_chance:
                sensation.pain = self.rng.triangular(0.0, 0.0, max_other_sensation)
            if self.rng.random() < self.sensation_activation_chance:
                sensation.pressure = self.rng.triangular(0.0, 0.0, max_other_sensation)
            if self.rng.random() < self.sensation_activation_chance:
                sensation.vibration = self.rng.triangular(0.0, 0.0, max_other_sensation)
            if self.rng.random() < self.sensation_activation_chance:
                sensation.tension = self.rng.triangular(0.0, 0.0, 1.0)
            if self.rng.random() < self.sensation_activation_chance:
                sensation.temperature = self.rng.triangular(-1.0, 0.0, 1.0)

            if any(val != 0.0 for val in asdict(sensation).values()):
                physical_state[part] = sensation

        return physical_state

    def generate(self) -> List[SensationVector]:
        """
        Generates the full list of SensationVector objects.
        """
        generated_vectors: List[SensationVector] = []

        for i in range(self.number_of_samples):
            params = {
                "valence": self.rng.triangular(-1.0, 0.0, 1.0),
                "arousal": self.rng.triangular(-1.0, 0.0, 1.0),
                "confidence": self.rng.triangular(-1.0, 0.0, 1.0),
                "focus": self.rng.triangular(-1.0, 0.0, 1.0),
                "openness": self.rng.triangular(-1.0, 0.0, 1.0),
                "dominance": self.rng.triangular(-1.0, 0.0, 1.0),
                "energy_level": self.rng.triangular(0.0, 0.9, 1.0),
            }

            if params["confidence"] < -0.5:
                params["dominance"] = min(params["dominance"], params["confidence"] + self.rng.uniform(0, 0.4))
            if params["energy_level"] > 0.8 and params["arousal"] < 0:
                params["arousal"] = max(params["arousal"], self.rng.uniform(0.1, 0.4))
            if params["energy_level"] < 0.1:
                params["focus"] = min(params["focus"], self.rng.uniform(-0.8, 0.0))

            sensation_vector = SensationVector(
                **params,
                physical_state=self._generate_physical_state()
            )

            generated_vectors.append(sensation_vector)

        return generated_vectors


if __name__ == "__main__":
    # --- Example Usage of the Updated Generator ---

    # We now set `max_active_body_parts` to 2.
    # The output will now have at most 2 body parts with sensations per vector.
    generator = SensationVectorGenerator(
        number_of_samples=10,
        max_active_body_parts=1,  # <-- New parameter in action
        sensation_activation_chance=0.4,
        seed=123
    )

    sensation_list = generator.generate()

    print(f"--- Generated {len(sensation_list)} Sensation Vectors (max 2 active body parts) ---\n")
    for index, vector in enumerate(sensation_list[:5]):
        print(f"--- Vector {index + 1} ---")
        print(vector)
        print("-" * 20 + "\n")