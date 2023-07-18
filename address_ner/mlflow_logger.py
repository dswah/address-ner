import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict
from functools import partial
import json

import numpy as np
import mlflow
import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

from address_ner.version import __version__


EXPERIMENT_NAME = "address-ner"


class DistilBERTNER(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        config_path = context.artifacts["config"]
        artifacts_dir = config_path.rsplit(r"/", 1)[0]
        tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
        model = AutoModelForTokenClassification.from_pretrained(artifacts_dir)
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)

        with Path(context.artifacts["metadata"]).open("rb") as fp:
            self.metadata = json.load(fp)

    @property
    def model_info(self):
        return {
            "model_run_id": self.metadata["run_id"],
            "train_code_version": self.metadata["train_code_version"],
            "inference_code_version": __version__,
            "model_class": self.__class__.__name__,
        }

    def predict(self, context, model_input):
        """predict single sample and send metadata

        Parameters
        ----------
        context : mlflow context - not used
        model_input : model_input : Dict with keys
            text : str

        Returns
        -------
        dict
        """
        text = model_input.get("text")

        assert isinstance(text, str)

        ents = self.nlp(text, aggregation_strategy="first")
        return {
            "prediction": ents,
            "model_info": self.model_info
        }



def dump_json(path, obj):
    with open(path, "w") as fp:
        json.dump(obj, fp)


def log(
    model: transformers.AutoModelForTokenClassification,
    tokenizer: transformers.AutoTokenizer,
    metrics: Dict[str, float]={},
    parameters: Dict[str, Any]={},
    experiment_name: str=EXPERIMENT_NAME,
    register: bool=False,
):

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        if parameters:
            mlflow.log_params(parameters)

        if metrics:
            # first eliminate arrays like confusion matrices
            array_metric_keys = []
            for metric in metrics:
                for name in ["confusion_matrix"]:
                    if name in metric:
                        array_metric_keys.append(metric)
            array_metrics = {}
            for metric in array_metric_keys:
                array_metrics[metric] = np.array(metrics.pop(metric))

            # log float metrics
            mlflow.log_metrics(metrics)

            # log array metrics as json dumps
            for metric in array_metrics:
                mlflow.log_dict(array_metrics[metric].tolist(), f"{metric}.json")

        metadata = {
            "run_id": run.info.run_id,
            "train_code_version": __version__
        }
        artifacts = {}
        with TemporaryDirectory() as td:

            model.save_pretrained(td)
            tokenizer.save_pretrained(td)

            with Path(td).joinpath("config.json").open() as fp:
                config = json.load(fp)

            config["_name_or_path"] = td

            with Path(td).joinpath("config.json").open("w") as fp:
                json.dump(config, fp)

            with Path(td).joinpath("tokenizer_config.json").open() as fp:
                config = json.load(fp)

            config["name_or_path"] = td

            with Path(td).joinpath("tokenizer_config.json").open("w") as fp:
                json.dump(config, fp)

            files = Path(td).glob("*")

            for f in files:
                artifacts[f.stem] = str(f)

            metadata_path = f"{td}/metadata.json"
            dump_json(metadata_path, metadata)
            artifacts["metadata"] = metadata_path

            mlflow.pyfunc.log_model(
                python_model=DistilBERTNER(),
                artifacts=artifacts,
                artifact_path="model", # location on remote mlflow
                pip_requirements=str(Path(__file__).parent.joinpath("requirements.txt")),
            )

        if register:
            model_uri = "runs:/{}/model".format(run.info.run_id)
            mlflow.register_model(model_uri, experiment_name)
