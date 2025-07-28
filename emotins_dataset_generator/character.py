
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import random






# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class CharacterTrait(BaseModel):
    """
    Represents a single, distinct trait of a character's personality.
    """
    name: str = Field(..., description="Unique identifier for the trait (e.g., 'courageous').")
    display_name: str = Field(..., description="The human-readable name of the character trait (e.g., 'Courageous').")
    description: str = Field(..., description="A clear description of what the trait entails.")
    antagonist: Optional[str] = Field(None, description="The key of the opposing character trait, if applicable.")


class Character(BaseModel):
    """
    Represents a character, defined by a name, a core description,
    and a collection of specific personality traits.
    """
    name: Optional[str] = Field(None, description="The full name of the character (optional).")
    description: Optional[str] = Field(None,
                                       description="A brief, evocative description of the character's essence (optional).")
    traits: List[CharacterTrait] = Field(..., description="A list of traits that define the character's personality.")




# ---------------------------------------------------------------------------
# TRAIT DEFINITIONS
# ---------------------------------------------------------------------------

# --- Core Moral Traits ---
TRAIT_COURAGEOUS = CharacterTrait(name="courageous", display_name="Courageous",
                                  description="Acts bravely in the face of fear, danger, or pain.",
                                  antagonist="cowardly")
TRAIT_COWARDLY = CharacterTrait(name="cowardly", display_name="Cowardly",
                                description="Shows a lack of bravery; is easily intimidated.", antagonist="courageous")
TRAIT_HONEST = CharacterTrait(name="honest", display_name="Honest",
                              description="Truthful and sincere; does not steal, cheat, or deceive.",
                              antagonist="deceitful")
TRAIT_DECEITFUL = CharacterTrait(name="deceitful", display_name="Deceitful",
                                 description="Uses deception and misleads others for personal gain.",
                                 antagonist="honest")
TRAIT_COMPASSIONATE = CharacterTrait(name="compassionate", display_name="Compassionate",
                                     description="Feels or shows sympathy and concern for others.", antagonist="cruel")
TRAIT_CRUEL = CharacterTrait(name="cruel", display_name="Cruel",
                             description="Willfully causes pain or suffering to others, or feels no concern about it.",
                             antagonist="compassionate")
TRAIT_PRINCIPLED = CharacterTrait(name="principled", display_name="Principled",
                                  description="Adheres firmly to a personal code of conduct or moral principles.",
                                  antagonist="unscrupulous")
TRAIT_UNSCRUPULOUS = CharacterTrait(name="unscrupulous", display_name="Unscrupulous",
                                    description="Has or shows no moral principles; not honest or fair.",
                                    antagonist="principled")
TRAIT_SELFLESS = CharacterTrait(name="selfless", display_name="Selfless",
                                description="Concerned more with the needs and wishes of others than with one's own.",
                                antagonist="selfish")
TRAIT_SELFISH = CharacterTrait(name="selfish", display_name="Selfish",
                               description="Lacks consideration for other people; is chiefly concerned with personal profit or pleasure.",
                               antagonist="selfless")

# --- Social & Interpersonal Traits ---
TRAIT_CHARISMATIC = CharacterTrait(name="charismatic", display_name="Charismatic",
                                   description="Exhibits a compelling charm that inspires devotion in others.",
                                   antagonist="awkward")
TRAIT_AWKWARD = CharacterTrait(name="awkward", display_name="Awkward",
                               description="Lacks social grace and is often clumsy or embarrassing in social situations.",
                               antagonist="charismatic")
TRAIT_MANIPULATIVE = CharacterTrait(name="manipulative", display_name="Manipulative",
                                    description="Controls or influences others in a clever or unscrupulous way.",
                                    antagonist="genuine")
TRAIT_GENUINE = CharacterTrait(name="genuine", display_name="Genuine",
                               description="Authentic and sincere; not pretending or false.", antagonist="manipulative")
TRAIT_LOYAL = CharacterTrait(name="loyal", display_name="Loyal",
                             description="Gives or shows firm and constant support or allegiance to a person or institution.",
                             antagonist="treacherous")
TRAIT_TREACHEROUS = CharacterTrait(name="treacherous", display_name="Treacherous",
                                   description="Guilty of or involving betrayal or deception.", antagonist="loyal")
TRAIT_INTROVERTED = CharacterTrait(name="introverted", display_name="Introverted",
                                   description="Prefers calm, minimally stimulating environments; recharges through solitude.",
                                   antagonist="extroverted")
TRAIT_EXTROVERTED = CharacterTrait(name="extroverted", display_name="Extroverted",
                                   description="Outgoing, overtly expressive, and energized by social interaction.",
                                   antagonist="introverted")

# --- Intellectual & Cognitive Traits ---
TRAIT_ANALYTICAL = CharacterTrait(name="analytical", display_name="Analytical",
                                  description="Excels at using logic and critical thinking to analyze situations.",
                                  antagonist="impulsive")
TRAIT_IMPULSIVE = CharacterTrait(name="impulsive", display_name="Impulsive",
                                 description="Acts on sudden urges or desires without forethought.",
                                 antagonist="analytical")
TRAIT_CREATIVE = CharacterTrait(name="creative", display_name="Creative",
                                description="Has the ability to generate or recognize novel and valuable ideas.",
                                antagonist="unimaginative")
TRAIT_UNIMAGINATIVE = CharacterTrait(name="unimaginative", display_name="Unimaginative",
                                     description="Lacks imagination or originality.", antagonist="creative")
TRAIT_PERCEPTIVE = CharacterTrait(name="perceptive", display_name="Perceptive",
                                  description="Shows keen insight and an intuitive understanding of people and situations.",
                                  antagonist="oblivious")
TRAIT_OBLIVIOUS = CharacterTrait(name="oblivious", display_name="Oblivious",
                                 description="Unaware of or not concerned about what is happening around them.",
                                 antagonist="perceptive")
TRAIT_CURIOUS = CharacterTrait(name="curious", display_name="Curious", description="Eager to know or learn something.",
                               antagonist="indifferent")
TRAIT_INDIFFERENT = CharacterTrait(name="indifferent", display_name="Indifferent",
                                   description="Has no particular interest or sympathy; unconcerned.",
                                   antagonist="curious")

# --- Temperament & Disposition ---
TRAIT_OPTIMISTIC = CharacterTrait(name="optimistic", display_name="Optimistic",
                                  description="Hopeful and confident about the future.", antagonist="pessimistic")
TRAIT_PESSIMISTIC = CharacterTrait(name="pessimistic", display_name="Pessimistic",
                                   description="Tends to see the worst aspect of things or believe that the worst will happen.",
                                   antagonist="optimistic")
TRAIT_CALM = CharacterTrait(name="calm", display_name="Calm",
                            description="Not showing or feeling nervousness, anger, or other strong emotions.",
                            antagonist="volatile")
TRAIT_VOLATILE = CharacterTrait(name="volatile", display_name="Volatile",
                                description="Liable to change rapidly and unpredictably, especially for the worse.",
                                antagonist="calm")
TRAIT_CYNICAL = CharacterTrait(name="cynical", display_name="Cynical",
                               description="Believes that people are motivated purely by self-interest; distrustful of human sincerity.",
                               antagonist="naive")
TRAIT_NAIVE = CharacterTrait(name="naive", display_name="Naive",
                             description="Shows a lack of experience, wisdom, or judgment; innocent.",
                             antagonist="cynical")
TRAIT_RESILIENT = CharacterTrait(name="resilient", display_name="Resilient",
                                 description="Able to withstand or recover quickly from difficult conditions.",
                                 antagonist="fragile")
TRAIT_FRAGILE = CharacterTrait(name="fragile", display_name="Fragile",
                               description="Easily broken or damaged emotionally or physically.",
                               antagonist="resilient")

# --- Work Ethic & Ambition ---
TRAIT_AMBITIOUS = CharacterTrait(name="ambitious", display_name="Ambitious",
                                 description="Has a strong desire and determination to achieve success.",
                                 antagonist="complacent")
TRAIT_COMPLACENT = CharacterTrait(name="complacent", display_name="Complacent",
                                  description="Shows smug or uncritical satisfaction with oneself or one's achievements.",
                                  antagonist="ambitious")
TRAIT_DILIGENT = CharacterTrait(name="diligent", display_name="Diligent",
                                description="Shows care and conscientiousness in one's work or duties.",
                                antagonist="lazy")
TRAIT_LAZY = CharacterTrait(name="lazy", display_name="Lazy",
                            description="Unwilling to use energy or effort; avoids exertion.", antagonist="diligent")
TRAIT_PRAGMATIC = CharacterTrait(name="pragmatic", display_name="Pragmatic",
                                 description="Deals with things sensibly and realistically in a way that is based on practical rather than theoretical considerations.",
                                 antagonist="idealistic")
TRAIT_IDEALISTIC = CharacterTrait(name="idealistic", display_name="Idealistic",
                                  description="Guided more by ideals than by practical considerations.",
                                  antagonist="pragmatic")

# ---------------------------------------------------------------------------
# TRAIT DICTIONARY HELPER
# ---------------------------------------------------------------------------

ALL_TRAITS: Dict[str, CharacterTrait] = {
    trait.name: trait for trait in [
        TRAIT_COURAGEOUS, TRAIT_COWARDLY, TRAIT_HONEST, TRAIT_DECEITFUL, TRAIT_COMPASSIONATE,
        TRAIT_CRUEL, TRAIT_PRINCIPLED, TRAIT_UNSCRUPULOUS, TRAIT_SELFLESS, TRAIT_SELFISH,
        TRAIT_CHARISMATIC, TRAIT_AWKWARD, TRAIT_MANIPULATIVE, TRAIT_GENUINE, TRAIT_LOYAL,
        TRAIT_TREACHEROUS, TRAIT_INTROVERTED, TRAIT_EXTROVERTED, TRAIT_ANALYTICAL, TRAIT_IMPULSIVE,
        TRAIT_CREATIVE, TRAIT_UNIMAGINATIVE, TRAIT_PERCEPTIVE, TRAIT_OBLIVIOUS, TRAIT_CURIOUS,
        TRAIT_INDIFFERENT, TRAIT_OPTIMISTIC, TRAIT_PESSIMISTIC, TRAIT_CALM, TRAIT_VOLATILE,
        TRAIT_CYNICAL, TRAIT_NAIVE, TRAIT_RESILIENT, TRAIT_FRAGILE, TRAIT_AMBITIOUS,
        TRAIT_COMPLACENT, TRAIT_DILIGENT, TRAIT_LAZY, TRAIT_PRAGMATIC, TRAIT_IDEALISTIC
    ]
}


# ---------------------------------------------------------------------------
# CHARACTER GENERATOR
# ---------------------------------------------------------------------------

def generate_character(
        num_traits: int,
        all_traits: Dict[str, CharacterTrait]
) -> Character:
    """
    Generates a character with a specified number of non-conflicting traits.

    Args:
        num_traits: The desired number of traits for the character.
        all_traits: A dictionary of all available character traits.

    Returns:
        A new Character instance with randomly selected traits.
    """
    if num_traits > len(all_traits) / 2:
        raise ValueError("Cannot select more than half of the total traits without guaranteed conflict.")

    selected_trait_objects: List[CharacterTrait] = []
    # Set to track keys of used traits and their antagonists to avoid contradiction
    used_keys: set[str] = set()
    # List of available trait keys to choose from
    available_keys: List[str] = list(all_traits.keys())
    random.shuffle(available_keys)  # Shuffle to ensure random selection

    for key in available_keys:
        if len(selected_trait_objects) >= num_traits:
            break

        # If the key has already been blocked (e.g., as an antagonist), skip it.
        if key in used_keys:
            continue

        # Get the trait object
        chosen_trait = all_traits[key]

        # Add the trait to our character
        selected_trait_objects.append(chosen_trait)

        # Block the chosen trait's key and its antagonist's key from future selection
        used_keys.add(chosen_trait.name)
        if chosen_trait.antagonist:
            used_keys.add(chosen_trait.antagonist)

    return Character(traits=selected_trait_objects)


# ---------------------------------------------------------------------------
# PRE-DEFINED CHARACTER EXAMPLES
# ---------------------------------------------------------------------------

# Example 1: The grizzled, by-the-book detective
DETECTIVE_KAITO = Character(
    name="Kaito Tanaka",
    description="A brilliant but world-weary detective who has seen the worst of humanity yet still clings to his own rigid code of justice.",
    traits=[
        TRAIT_ANALYTICAL,
        TRAIT_PERCEPTIVE,
        TRAIT_CYNICAL,
        TRAIT_PRINCIPLED,
        TRAIT_INTROVERTED,
        TRAIT_DILIGENT
    ]
)

# Example 2: The charming and ruthless diplomat
DIPLOMAT_ISABELLA = Character(
    name="Isabella Rossi",
    description="A master of statecraft who can charm a room into submission or orchestrate a political coup with equal ease. Her only allegiance is to power.",
    traits=[
        TRAIT_CHARISMATIC,
        TRAIT_AMBITIOUS,
        TRAIT_MANIPULATIVE,
        TRAIT_UNSCRUPULOUS,
        TRAIT_PRAGMATIC,
        TRAIT_EXTROVERTED
    ]
)

# Example 3: The pure-hearted, determined healer
HEALER_ELARA = Character(
    name="Elara",
    description="A young healer whose compassion is her greatest strength and most exploitable weakness. She believes in the inherent goodness of all, even when faced with evidence to the contrary.",
    traits=[
        TRAIT_COMPASSIONATE,
        TRAIT_IDEALISTIC,
        TRAIT_SELFLESS,
        TRAIT_NAIVE,
        TRAIT_RESILIENT,
        TRAIT_COURAGEOUS
    ]
)

# ---------------------------------------------------------------------------
# GENERATOR USAGE EXAMPLE
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Example: Generate a character with 5 random, non-conflicting traits
    random_character = generate_character(num_traits=5, all_traits=ALL_TRAITS)

    print("--- Generated Character ---")
    if random_character.name:
        print(f"Name: {random_character.name}")
    print(f"Number of Traits: {len(random_character.traits)}")
    for trait in random_character.traits:
        print(f"- {trait.display_name}: ({trait.name})")

