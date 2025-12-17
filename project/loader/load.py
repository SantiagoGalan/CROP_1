import os
import os
from keras.models import load_model
from project.custom_layers.sampling import Sampling
from project.custom_layers.reshapeLayer import ReshapeLayer
from project.data.get_data import get_mnist_data
from project.models_definitions.cvae import CVAE

BASE_DIR = os.path.dirname(__file__)          # project/trained_models/
COMMON_PATH = "project/trained_models/"

COMMON_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "trained_models")
)


def cvae(lat, inter, dataset):
    encoder_path = os.path.join(COMMON_PATH, "encoders", f"en_int_{inter}_lat_{lat}_{dataset}.keras")
    decoder_path = os.path.join(COMMON_PATH, "decoders", f"de_int_{inter}_lat_{lat}_{dataset}.keras")

    encoder = load_model(
        encoder_path,
        custom_objects={"Sampling": Sampling},
    )
    decoder = load_model(decoder_path)

    return CVAE(encoder, decoder,original_dim=28*28)



def data(dataset):
    return get_mnist_data(dataset=dataset)


def predictor(dataset):
    model_path = os.path.join(COMMON_PATH, "predictores", f"CCE_Conv2D_{dataset}.keras")
    return load_model(model_path, {"ReshapeLayer": ReshapeLayer})

def all_models(dataset):
    import os
    from keras.models import load_model
    from project.custom_layers.sampling import Sampling
    from project.models_definitions.cvae import CVAE

    encoders_dir = os.path.join(COMMON_PATH, "encoders")
    decoders_dir = os.path.join(COMMON_PATH, "decoders")

    encoder_files = sorted(os.listdir(encoders_dir))
    decoder_files = sorted(os.listdir(decoders_dir))

    def get_key(filename):
        return "_".join(filename.split("_")[2:])  

    encoders = {
        get_key(f): os.path.join(encoders_dir, f)
        for f in encoder_files
        if f.endswith(f"{dataset}.keras")
    }

    decoders = {
        get_key(f): os.path.join(decoders_dir, f)
        for f in decoder_files
        if f.endswith(f"{dataset}.keras")
    }

    common_keys = sorted(set(encoders.keys()) & set(decoders.keys()))
    print(f"Encontrados {len(common_keys)} pares de modelos.")

    models = []

    for key in common_keys:
        encoder_path = encoders[key]
        decoder_path = decoders[key]

        encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
        decoder = load_model(decoder_path)

        name = f"cvae_{key}"

        cvae = CVAE(encoder, decoder, original_dim=28*28, name=name)
        cvae.compile(optimizer="adam")

        models.append(cvae)
    return models