import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from keras.models import load_model
from custom_layers.Sampling import Sampling
from custom_layers.ReshapeLayer import ReshapeLayer
from data.get_data import get_mnist_data
from models_definitions.cvae import CVAE

COMMON_PATH = "../../trained_models"
def cvae(lat, inter, dataset):
    """
    Inputs: - Models path
            - Data name ("" for mnist "fashion" for fashion mnist)
    Outputs:- Train models
            - Dataset
    """
  
    encoder = load_model(
        f"{COMMON_PATH}/encoders/en_int_{inter}_lat_{lat}_{dataset}.keras",
        custom_objects={"Sampling": Sampling},
    )
    decoder = load_model(
        f"{COMMON_PATH}/decoders/de_int_{inter}_lat_{lat}_{dataset}.keras"
    )

    cvae = CVAE(encoder=encoder, decoder=decoder, original_dim=28 * 28, beta=1)
    # cvae.compile(optimizer="adam")
    return cvae


def data(dataset):
    return get_mnist_data(dataset=dataset)


def predictor(dataset):

    return load_model(
        f"{COMMON_PATH}/predictores/CCE_Conv2D_{dataset}.keras", {"ReshapeLayer": ReshapeLayer}
    )


def all_models(
    encoders_paths=f"{COMMON_PATH}/encoders/",
    decoders_paths=f"{COMMON_PATH}/decoders/",
):
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
    from keras.models import load_model
    from custom_layers.Sampling import Sampling
    from models_definitions.cvae import CVAE

    # Obtener listas de archivos
    encoder_files = sorted(os.listdir(encoders_paths))
    decoder_files = sorted(os.listdir(decoders_paths))

    # Función para extraer clave
    def get_key(filename):
        return "_".join(filename.split("_")[2:])

    # Crear diccionarios clave → path
    encoders = {
        get_key(f): os.path.join(encoders_paths, f)
        for f in encoder_files
        if f.endswith(".keras")
    }
    decoders = {
        get_key(f): os.path.join(decoders_paths, f)
        for f in decoder_files
        if f.endswith(".keras")
    }

    # Claves comunes entre encoder y decoder
    common_keys = sorted(set(encoders.keys()) & set(decoders.keys()))
    print(f"Encontrados {len(common_keys)} pares de modelos.")

    models = []

    # Iterar sobre modelos
    for key in common_keys:

        encoder_path = encoders[key]
        decoder_path = decoders[key]

        encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
        decoder = load_model(decoder_path)

        cvae = CVAE(encoder, decoder, original_dim=28 * 28)
        cvae.compile(optimizer="adam")

        models.append(cvae)

    return models
