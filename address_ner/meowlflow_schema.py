from typing import List, Text, Tuple, Dict
import json
from pathlib import Path

from pydantic import BaseModel

from meowlflow.api.base import BaseRequest, BaseResponse


title = "Address Named Entity Recognition"

description = "Model to identify addresses within legal texts. "\
"The model accepts one text at a time. Each request should contain a single string."\
"The service returns a list of address entities found within the text and "\
"their probabilities, as well as medatada about the model."


class EntityPrediction(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class Request(BaseRequest):
    text: str

    def transform(self):
        return self.dict()

    class Config:
        schema_extra = {
            "example": {
                "text": "16: Freiraurn, 547,68- 547,68 \n2 \n1 St \nÜbertrag: 988,00 \nwindmühlenweg 8 02625 bautzen sachs Firmensitz : Telefon : 03 40 / 5 40 09 - 0"
            }
        }


class Response(BaseResponse):
    prediction: List[EntityPrediction]
    model_info: Dict[str, str]

    @classmethod
    def transform(cls, data):
        return data


    class Config:
        schema_extra = {"example":
        {
  "prediction": [
    {
      "entity_group": "ADDRESS",
      "score": 0.9999479651451111,
      "word": "windmühlenweg 8 02625 bautzen",
      "start": 58,
      "end": 87
    }
  ],
  "model_info": {
    "model_run_id": "08aa164091a54dadb8b94ca21e8fe2e0",
    "train_code_version": "0.0.0.post70+e97fad6",
    "inference_code_version": "0.0.0.post70+e97fad6",
    "model_class": "DistilBERTNER"
  }
}
}
