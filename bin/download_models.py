from transformers import RobertaTokenizer
from transferprediction.huggingface_extensions import (
    RobertaForSequenceClassificationGLUE,
)

model = RobertaForSequenceClassificationGLUE.from_pretrained(
    "distilroberta-base"
)
tok = RobertaTokenizer.from_pretrained("distilroberta-base")
