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


def encoder(lat=None, inter=None, dataset=None, name=None):
    if name is not None:
        encoder_path = os.path.join(
            COMMON_PATH, "encoders", f"{name}.keras"
        )
    else:
        encoder_path = os.path.join(
            COMMON_PATH,
            "encoders",
            f"en_int_{inter}_lat_{lat}_{dataset}.keras"
        )

    return load_model(
        encoder_path,
        custom_objects={"Sampling": Sampling},
    )


def decoder(lat=None, inter=None, dataset=None, name=None):
    if name is not None:
        decoder_path = os.path.join(
            COMMON_PATH, "decoders", f"{name}.keras"
        )
    else:
        decoder_path = os.path.join(
            COMMON_PATH,
            "decoders",
            f"de_int_{inter}_lat_{lat}_{dataset}.keras"
        )

    return load_model(decoder_path)



def cvae(
    lat=None,
    inter=None,
    dataset=None,
    encoder_name=None,
    decoder_name=None,
    model_name=None,
):


    enc_dir = os.path.join(COMMON_PATH, "encoders")
    dec_dir = os.path.join(COMMON_PATH, "decoders")

    # Caso 1: se pasa un nombre base común
    if model_name is not None:
        if model_name.startswith("en_") or model_name.startswith("de_"):
            raise ValueError(
                "model_name debe ser el nombre base SIN 'en_' ni 'de_'"
            )

        encoder_name = f"en_{model_name}"
        decoder_name = f"de_{model_name}"

    # Caso 2: se pasan nombres explícitos
    if encoder_name is not None and decoder_name is not None:
        encoder_path = os.path.join(enc_dir, encoder_name)
        decoder_path = os.path.join(dec_dir, decoder_name)

    # Caso 3: modo original (lat, inter, dataset)
    elif lat is not None and inter is not None and dataset is not None:
        encoder_path = os.path.join(
            enc_dir, f"en_int_{inter}_lat_{lat}_{dataset}.keras"
        )
        decoder_path = os.path.join(
            dec_dir, f"de_int_{inter}_lat_{lat}_{dataset}.keras"
        )

    else:
        raise ValueError(
            "Debes pasar (lat, inter, dataset) "
            "o (encoder_name y decoder_name) "
            "o model_name"
        )

    encoder = load_model(
        encoder_path,
        custom_objects={"Sampling": Sampling},
    )
    decoder = load_model(decoder_path)

    return CVAE(encoder, decoder, original_dim=28 * 28)


def data(dataset):
    return get_mnist_data(dataset=dataset)


def predictor(
    dataset=None,
    model_name=None,
    model_path=None,
):
    """
    Carga un modelo predictor a partir de:
    - dataset (modo original),
    - model_name (nombre del archivo .keras),
    - model_path (ruta absoluta o relativa).
    """

    pred_dir = os.path.join(COMMON_PATH, "predictores")

    # Caso 1: ruta directa
    if model_path is not None:
        path = model_path

    # Caso 2: nombre explícito del modelo
    elif model_name is not None:
        path = os.path.join(pred_dir, model_name)

    # Caso 3: comportamiento original por dataset
    elif dataset is not None:
        if dataset == "early_stop_fashion":
            path = os.path.join(pred_dir, "early_stop_fashion.keras")
        else:
            path = os.path.join(pred_dir, f"CCE_Conv2D_{dataset}.keras")

    else:
        raise ValueError(
            "Debes pasar dataset, model_name o model_path"
        )

    return load_model(path, custom_objects={"ReshapeLayer": ReshapeLayer})


def parse_dims_from_key(key):
    parts = key.split("_")
    if len(parts) < 4 or parts[1] != "lat":
        raise ValueError(f"Formato de key inválido: {key}")

    int_dim = int(parts[0])
    lat_dim = int(parts[2])

    return int_dim, lat_dim

def all_models(dataset, lat=None, inter=None):
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

    # ---- FILTRADO POR inter / lat ----
    filtered_keys = []
    for key in common_keys:
        int_dim, lat_dim = parse_dims_from_key(key)

        if inter is not None and int_dim != inter:
            continue
        if lat is not None and lat_dim != lat:
            continue

        filtered_keys.append(key)

    if inter is not None and lat is not None and not filtered_keys:
        raise ValueError(
            f"No existe ningún modelo con inter={inter} y lat={lat} para dataset '{dataset}'"
        )

    print(f"Encontrados {len(filtered_keys)} pares de modelos.")
    models = []

    for key in common_keys:
        encoder_path = encoders[key]
        decoder_path = decoders[key]

        encoder = load_model(encoder_path, custom_objects={"Sampling": Sampling})
        decoder = load_model(decoder_path)

        int_dim, lat_dim = parse_dims_from_key(key)

        name = f"cvae_int_{int_dim}_lat_{lat_dim}"

        cvae = CVAE(
            encoder,
            decoder,
            original_dim=28*28,
            name=name
        )
        cvae.compile(optimizer="adam")

        models.append(cvae)

    return models
