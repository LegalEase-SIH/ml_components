import torch


MAX_LEN = 128
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.has_cuda else "mps" if torch.has_mps else "cpu")
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 64
HIDDEN_SIZE = 128


ENTITIES = [
    "WITNESS",
    "OTHER_PERSON",
    "STATUTE",
    "CASE_NUMBER",
    "GPE",
    "ORG",
    "DATE",
    "JUDGE",
    "PROVISION",
    "PETITIONER",
    "RESPONDENT",
    "COURT",
    "PRECEDENT",
]

NUM_TAGS = len(ENTITIES)
