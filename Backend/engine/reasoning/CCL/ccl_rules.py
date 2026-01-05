# ccl_rules.py
# Observe-only rules for CCL v1

DOMAIN_NORMALIZATION_MAP = {
    "astronomy_physics": "Physical Reality",
    "chemistry": "Physical Reality",
    "biology_medicine": "Biological / Medical Reality",
    "economics": "Socioeconomic Systems",
    "history_society": "Historical / Social Systems",
    "general_knowledge": "General Knowledge",
}

DOMAIN_KEYWORDS = {
    "astronomy_physics": [
        "sun", "moon", "earth", "planet", "orbit", "gravity", "space", "universe"
    ],
    "biology_medicine": [
        "human", "body", "oxygen", "breathe", "disease", "virus",
        "health", "medicine", "mental", "stress"
    ],
    "chemistry": [
        "water", "fire", "chemical", "compound", "element", "reaction"
    ],
    "history_society": [
        "history", "ancient", "medieval", "government", "policy", "law"
    ],
    "economics": [
        "economy", "economic", "inflation", "market", "finance"
    ],
}

ABSTRACT_ENTITIES = [
    "economy", "market", "society", "democracy",
    "system", "nation", "culture", "government"
]

PHYSICAL_ACTIONS = [
    "bleeding", "dying", "dead", "eating",
    "choking", "burning", "freezing", "collapsing"
]
