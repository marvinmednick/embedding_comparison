from bigml.topicmodel import TopicModel
import json
import os
import gzip
from log_setup import get_logger
from utils import shorten_text
from polls.models import Embedding32

logger = get_logger(__name__)

MODEL_DIR = "./topic_models"
MAX_STRING_LENGTH = 128000

models_file = os.path.join(MODEL_DIR, "models.json")
with open(models_file, "r") as fin:
    GLOBAL_MODELS = json.load(fin)

TOP_LEVEL_MODELS = GLOBAL_MODELS["top_level_models"]
SUBTOPIC_MODELS = GLOBAL_MODELS["subtopic_models"]

TOP_LEVEL_SHORT_NAME = {
    "Automotive Technology": "at",
    "Blockchain and Decentralized Technology": "bc",
    "CRISPR": "crips",
    "Computing Technology": "compute",
    "Content Protection": "content_protect",
    "Digital Audio Technology": "dat",
    "Digital Video Technology": "dvt",
    "Display Technology": "display",
    "Electric Automotive Vehicles": "ev_auto",
    "Electric Vehicles": "ev",
    "Gene-based Technology": "gene",
    "Health Technology": "health",
    "Internet Technology": "internet",
    "Internet of Things": "inet_things",
    "IoT": "iot",
    "Location Technology": "location",
    "Mobile Devices": "mobile",
    "Optical Technology": "optical",
    "Security Technology": "security",
    "Telecommunications and Cellular Networks": "telecom",
    "Video Compression": "vc",
    "Virtual Reality": "vr",
    "Wired Interconnect Technology": "wired_inet",
    "Wireless Networking": "wireless"
}


def display_models():

    print(f"{'Model Name':<50} {'Short Name':<20}")
    for model, short_name in TOP_LEVEL_SHORT_NAME.items():
        print(f"{model:<50} {short_name:<20}")


def get_model_from_file(model_id):

    model_file_name = model_id.replace('/', '_') + '.gz'
    model_path = os.path.join(MODEL_DIR, model_file_name)

    with gzip.open(model_path, 'rb') as f:
        file_content = f.read()

    json_data = json.loads(file_content)

    return TopicModel(json_data)


def get_full_model_name(model_name) -> str:
    if not model_name:
        return None

    if model_name in TOP_LEVEL_SHORT_NAME:
        return model_name
    else:
        for key, value in TOP_LEVEL_SHORT_NAME.items():
            if model_name == value:
                return key
        return None


def concatenate_sections(section_list):
    total_remaining = MAX_STRING_LENGTH
    sections_remaining = len(section_list)
    shortened = []

    for section in sorted(section_list, key=lambda x: len(x)):
        max_length = total_remaining // sections_remaining
        sample = shorten_text(section, max_length)
        shortened.append(sample)

        total_remaining -= len(sample)
        sections_remaining -= 1

    return " ".join([s for s in shortened])


class TopicModelEmbedding():
    def __init__(self, domain, document):
        self._domain = domain
        self.select_subtype_model(document)
        logger.info("Using Topic Model %s.  Topic %s  short name %s  Model name: %s",
                    self.domain,
                    self._topic,
                    self._topic_model_short_name,
                    self._model_name)

    def generate_embedding(self, document: str) -> list[float]:
        result = self._model.distribution({"Text": document})
        return [x['probability'] for x in result]

    def chunk(self, document):
        return [document], 1

    def select_subtype_model(self, sections):
        domain_model_name = TOP_LEVEL_MODELS[self._domain]
        self._domain_model_name = domain_model_name

        domain_model = get_model_from_file(domain_model_name)

        document = concatenate_sections(sections)
        dist = domain_model.distribution({"Text": document})
        self._topic = sorted(dist, key=lambda x: -x["probability"])[0]["name"]

        self._model_name = SUBTOPIC_MODELS[self._domain][self._topic]
        self._model = get_model_from_file(self._model_name)

        self._model_short_name = TOP_LEVEL_SHORT_NAME[self._domain]
        self._topic_model_short_name = f"tpm_{self._model_short_name}_{self._topic}"

    def get_embed_params(self):
        lookup_params = {
            'name': f"TOPIC_MODEL_{self._domain.upper()} - {self._topic}",
        }

        defaults = {
            'size': 32,
            'short_name': self._topic_model_short_name,
            'description': "Topic Model in the {self._domain.upper() and topic {self._topic}"
        }
        model = Embedding32

        return lookup_params, defaults, model

    @property
    def topic(self):
        return self._topic

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_short_name(self):
        return self._model_short_name

    @property
    def domain(self):
        return self._domain

    @property
    def domain_model_name(self):
        return self._domain_model_name
