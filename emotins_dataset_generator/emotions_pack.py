from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class Emotion(BaseModel):
    """
    Represents an emotion with its name and potential antagonists.
    """
    name: str = Field(..., description="The name of the emotion (e.g., 'joy', 'sadness').")
    antagonist: Optional[List[str]] = Field(None, description="A list of antagonistic emotion keys.")

# ---------------------------------------------------------------------------
# EMOTION DEFINITIONS
# ---------------------------------------------------------------------------

# An optimized and comprehensive dictionary of human emotions.
AVAILABLE_EMOTIONS_NAME: Dict[str, str] = {
    # --- Positive: High Arousal (6) ---
    "joy": "Joy", "excitement": "Excitement", "ecstasy": "Ecstasy",
    "enthusiasm": "Enthusiasm", "thrill": "Thrill", "glee": "Glee",
    # --- Positive: Low Arousal (7) ---
    "happiness": "Happiness", "pleasure": "Pleasure", "satisfaction": "Satisfaction",
    "serenity": "Serenity", "calmness": "Calmness", "relaxation": "Relaxation",
    "relief": "Relief",
    # --- Positive: Social & Cognitive (13) ---
    "love": "Love", "affection": "Affection", "admiration": "Admiration",
    "trust": "Trust", "pride": "Pride", "gratitude": "Gratitude",
    "hope": "Hope", "optimism": "Optimism", "curiosity": "Curiosity",
    "awe": "Awe", "inspiration": "Inspiration", "moved": "Moved",
    "amusement": "Amusement",
    # --- Desire, Attraction & Intimacy (8) ---
    "lust": "Lust", "passion": "Passion", "arousal": "Arousal",
    "attraction": "Attraction", "infatuation": "Infatuation", "yearning": "Yearning",
    "temptation": "Temptation", "intimacy": "Intimacy",
    # --- Negative: Anger Family (7) ---
    "anger": "Anger", "rage": "Rage", "annoyance": "Annoyance",
    "frustration": "Frustration", "resentment": "Resentment", "indignation": "Indignation",
    "contempt": "Contempt",
    # --- Negative: Sadness Family (9) ---
    "sadness": "Sadness", "grief": "Grief", "despair": "Despair",
    "melancholy": "Melancholy", "disappointment": "Disappointment", "loneliness": "Loneliness",
    "misery": "Misery", "pity": "Pity", "guilt": "Guilt",
    # --- Negative: Fear Family (7) ---
    "fear": "Fear", "horror": "Horror", "anxiety": "Anxiety",
    "worry": "Worry", "nervousness": "Nervousness", "dread": "Dread",
    "panic": "Panic",
    # --- Negative: Aversion & Social (10) ---
    "disgust": "Disgust", "shame": "Shame", "embarrassment": "Embarrassment",
    "humiliation": "Humiliation", "envy": "Envy", "jealousy": "Jealousy",
    "insecurity": "Insecurity", "doubt": "Doubt", "boredom": "Boredom",
    "weariness": "Weariness",
    # --- Surprise Family (3) ---
    "surprise": "Surprise", "astonishment": "Astonishment", "shock": "Shock",
}

# ---------------------------------------------------------------------------
# EMOTION INSTANCES
# ---------------------------------------------------------------------------

# --- Positive: High Arousal ---
JOY = Emotion(name="joy", antagonist=["sadness", "grief", "misery"])
EXCITEMENT = Emotion(name="excitement", antagonist=["calmness", "boredom", "weariness"])
ECSTASY = Emotion(name="ecstasy", antagonist=["despair", "misery"])
ENTHUSIASM = Emotion(name="enthusiasm", antagonist=["boredom", "weariness"])
THRILL = Emotion(name="thrill", antagonist=["boredom", "relaxation"])
GLEE = Emotion(name="glee", antagonist=["pity", "shame"])

# --- Positive: Low Arousal ---
HAPPINESS = Emotion(name="happiness", antagonist=["sadness", "misery"])
PLEASURE = Emotion(name="pleasure", antagonist=["disgust", "annoyance"])
SATISFACTION = Emotion(name="satisfaction", antagonist=["frustration", "disappointment"])
SERENITY = Emotion(name="serenity", antagonist=["anxiety", "rage"])
CALMNESS = Emotion(name="calmness", antagonist=["anger", "panic", "excitement"])
RELAXATION = Emotion(name="relaxation", antagonist=["nervousness", "anxiety"])
RELIEF = Emotion(name="relief", antagonist=["dread", "worry"])

# --- Positive: Social & Cognitive ---
LOVE = Emotion(name="love", antagonist=["contempt", "resentment"])
AFFECTION = Emotion(name="affection", antagonist=["disgust", "contempt"])
ADMIRATION = Emotion(name="admiration", antagonist=["contempt", "envy"])
TRUST = Emotion(name="trust", antagonist=["doubt", "insecurity"])
PRIDE = Emotion(name="pride", antagonist=["shame", "humiliation"])
GRATITUDE = Emotion(name="gratitude", antagonist=["resentment", "envy"])
HOPE = Emotion(name="hope", antagonist=["despair", "dread"])
OPTIMISM = Emotion(name="optimism", antagonist=["melancholy"])
CURIOSITY = Emotion(name="curiosity", antagonist=["boredom"])
AWE = Emotion(name="awe", antagonist=["boredom", "contempt"])
INSPIRATION = Emotion(name="inspiration", antagonist=["weariness", "frustration"])
MOVED = Emotion(name="moved", antagonist=["contempt", "boredom"])
AMUSEMENT = Emotion(name="amusement", antagonist=["boredom"])

# --- Desire, Attraction & Intimacy ---
LUST = Emotion(name="lust", antagonist=["disgust", "shame"])
PASSION = Emotion(name="passion", antagonist=["boredom", "weariness"])
AROUSAL = Emotion(name="arousal", antagonist=["disgust", "calmness"])
ATTRACTION = Emotion(name="attraction", antagonist=["disgust"])
INFATUATION = Emotion(name="infatuation", antagonist=["contempt"])
YEARNING = Emotion(name="yearning", antagonist=["satisfaction", "relief"])
TEMPTATION = Emotion(name="temptation", antagonist=["guilt", "disgust"])
INTIMACY = Emotion(name="intimacy", antagonist=["loneliness", "insecurity"])

# --- Negative: Anger Family ---
ANGER = Emotion(name="anger", antagonist=["calmness", "fear"])
RAGE = Emotion(name="rage", antagonist=["serenity", "calmness"])
ANNOYANCE = Emotion(name="annoyance", antagonist=["pleasure", "satisfaction"])
FRUSTRATION = Emotion(name="frustration", antagonist=["satisfaction", "relief"])
RESENTMENT = Emotion(name="resentment", antagonist=["gratitude", "affection"])
INDIGNATION = Emotion(name="indignation", antagonist=["admiration", "trust"])
CONTEMPT = Emotion(name="contempt", antagonist=["admiration", "love"])

# --- Negative: Sadness Family ---
SADNESS = Emotion(name="sadness", antagonist=["joy", "happiness"])
GRIEF = Emotion(name="grief", antagonist=["joy"])
DESPAIR = Emotion(name="despair", antagonist=["hope", "ecstasy"])
MELANCHOLY = Emotion(name="melancholy", antagonist=["optimism", "joy"])
DISAPPOINTMENT = Emotion(name="disappointment", antagonist=["satisfaction"])
LONELINESS = Emotion(name="loneliness", antagonist=["intimacy", "affection"])
MISERY = Emotion(name="misery", antagonist=["happiness", "ecstasy"])
PITY = Emotion(name="pity", antagonist=["glee", "admiration"])
GUILT = Emotion(name="guilt", antagonist=["pride", "temptation"])

# --- Negative: Fear Family ---
FEAR = Emotion(name="fear", antagonist=["anger", "trust", "calmness"])
HORROR = Emotion(name="horror", antagonist=["serenity", "awe"])
ANXIETY = Emotion(name="anxiety", antagonist=["calmness", "serenity"])
WORRY = Emotion(name="worry", antagonist=["relief", "trust"])
NERVOUSNESS = Emotion(name="nervousness", antagonist=["relaxation", "calmness"])
DREAD = Emotion(name="dread", antagonist=["hope", "relief"])
PANIC = Emotion(name="panic", antagonist=["calmness", "serenity"])

# --- Negative: Aversion & Social ---
DISGUST = Emotion(name="disgust", antagonist=["pleasure", "attraction", "lust"])
SHAME = Emotion(name="shame", antagonist=["pride"])
EMBARRASSMENT = Emotion(name="embarrassment", antagonist=["pride"])
HUMILIATION = Emotion(name="humiliation", antagonist=["pride", "admiration"])
ENVY = Emotion(name="envy", antagonist=["gratitude", "admiration"])
JEALOUSY = Emotion(name="jealousy", antagonist=["trust", "admiration"])
INSECURITY = Emotion(name="insecurity", antagonist=["trust", "pride"])
DOUBT = Emotion(name="doubt", antagonist=["trust"])
BOREDOM = Emotion(name="boredom", antagonist=["excitement", "curiosity", "enthusiasm"])
WEARINESS = Emotion(name="weariness", antagonist=["enthusiasm", "excitement", "inspiration"])

# --- Surprise Family ---
SURPRISE = Emotion(name="surprise", antagonist=["boredom"]) # No perfect antagonist, 'anticipation' is best but not in list
ASTONISHMENT = Emotion(name="astonishment", antagonist=["boredom"])
SHOCK = Emotion(name="shock", antagonist=["calmness", "serenity"])


# ---------------------------------------------------------------------------
# FINAL EMOTIONS DICTIONARY
# ---------------------------------------------------------------------------

EMOTIONS: Dict[str, Emotion] = {
    "joy": JOY, "excitement": EXCITEMENT, "ecstasy": ECSTASY, "enthusiasm": ENTHUSIASM,
    "thrill": THRILL, "glee": GLEE, "happiness": HAPPINESS, "pleasure": PLEASURE,
    "satisfaction": SATISFACTION, "serenity": SERENITY, "calmness": CALMNESS,
    "relaxation": RELAXATION, "relief": RELIEF, "love": LOVE, "affection": AFFECTION,
    "admiration": ADMIRATION, "trust": TRUST, "pride": PRIDE, "gratitude": GRATITUDE,
    "hope": HOPE, "optimism": OPTIMISM, "curiosity": CURIOSITY, "awe": AWE,
    "inspiration": INSPIRATION, "moved": MOVED, "amusement": AMUSEMENT, "lust": LUST,
    "passion": PASSION, "arousal": AROUSAL, "attraction": ATTRACTION,
    "infatuation": INFATUATION, "yearning": YEARNING, "temptation": TEMPTATION,
    "intimacy": INTIMACY, "anger": ANGER, "rage": RAGE, "annoyance": ANNOYANCE,
    "frustration": FRUSTRATION, "resentment": RESENTMENT, "indignation": INDIGNATION,
    "contempt": CONTEMPT, "sadness": SADNESS, "grief": GRIEF, "despair": DESPAIR,
    "melancholy": MELANCHOLY, "disappointment": DISAPPOINTMENT, "loneliness": LONELINESS,
    "misery": MISERY, "pity": PITY, "guilt": GUILT, "fear": FEAR, "horror": HORROR,
    "anxiety": ANXIETY, "worry": WORRY, "nervousness": NERVOUSNESS, "dread": DREAD,
    "panic": PANIC, "disgust": DISGUST, "shame": SHAME, "embarrassment": EMBARRASSMENT,
    "humiliation": HUMILIATION, "envy": ENVY, "jealousy": JEALOUSY,
    "insecurity": INSECURITY, "doubt": DOUBT, "boredom": BOREDOM, "weariness": WEARINESS,
    "surprise": SURPRISE, "astonishment": ASTONISHMENT, "shock": SHOCK,
}
