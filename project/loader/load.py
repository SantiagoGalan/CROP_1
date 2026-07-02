import os
from keras.models import load_model

from project.custom_layers.sampling import Sampling
from project.custom_layers.reshapeLayer import ReshapeLayer
from project.data.get_data import get_mnist_data
from project.models_definitions.cvae import CVAE
from project.models_definitions.decoder import Decoder
from project.models_definitions.encoder import Encoder


class Loader:
    # ---------- paths (class-level) ----------

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "trained_models")
    )

    ENCODERS_DIR = os.path.join(BASE_DIR, "encoders")
    DECODERS_DIR = os.path.join(BASE_DIR, "decoders")
    PREDICTORS_DIR = os.path.join(BASE_DIR, "predictores")

    # ---------- basic loaders ----------

    @staticmethod
    def _dims_from_model_path(path):
        filename = os.path.basename(path)
        parts = filename.split("_")
        try:
            int_dim = int(parts[parts.index("int") + 1])
            lat_dim = int(parts[parts.index("lat") + 1])
        except (ValueError, IndexError):
            int_dim = None
            lat_dim = None

        return int_dim, lat_dim

    @classmethod
    def _encoder_custom_objects(cls, path=None):
        int_dim, lat_dim = cls._dims_from_model_path(path or "")

        if int_dim is None or lat_dim is None:
            encoder_cls = Encoder
        else:
            class LegacyEncoder(Encoder):
                def __init__(self, **kwargs):
                    kwargs.pop("intermediate_dim", None)
                    kwargs.pop("latent_dim", None)
                    super().__init__(
                        intermediate_dim=int_dim,
                        latent_dim=lat_dim,
                        **kwargs
                    )

            encoder_cls = LegacyEncoder

        return {
            "Sampling": Sampling,
            "Custom>Sampling": Sampling,
            "Encoder": encoder_cls,
            "Custom>Encoder": encoder_cls,
        }

    @classmethod
    def _decoder_custom_objects(cls, path=None):
        int_dim, lat_dim = cls._dims_from_model_path(path or "")

        if int_dim is None or lat_dim is None:
            decoder_cls = Decoder
        else:
            class LegacyDecoder(Decoder):
                def __init__(self, **kwargs):
                    kwargs.pop("intermediate_dim", None)
                    kwargs.pop("latent_dim", None)
                    super().__init__(
                        intermediate_dim=int_dim,
                        latent_dim=lat_dim,
                        **kwargs
                    )

            decoder_cls = LegacyDecoder

        return {
            "Decoder": decoder_cls,
            "Custom>Decoder": decoder_cls,
        }

    @classmethod
    def encoder(cls, lat=None, inter=None, dataset=None, name=None):
        if name is not None:
            path = os.path.join(cls.ENCODERS_DIR, f"{name}.keras")
        else:
            path = os.path.join(
                cls.ENCODERS_DIR,
                f"en_int_{inter}_lat_{lat}_{dataset}.keras"
            )

        return load_model(path, custom_objects=cls._encoder_custom_objects(path))

    @classmethod
    def decoder(cls, lat=None, inter=None, dataset=None, name=None):
        if name is not None:
            path = os.path.join(cls.DECODERS_DIR, f"{name}.keras")
        else:
            path = os.path.join(
                cls.DECODERS_DIR,
                f"de_int_{inter}_lat_{lat}_{dataset}.keras"
            )

        return load_model(path, custom_objects=cls._decoder_custom_objects(path))

    # ---------- CVAE ----------

    @classmethod
    def cvae(
        cls,
        lat=None,
        inter=None,
        dataset=None,
        encoder_name=None,
        decoder_name=None,
        model_name=None,
    ):
        if model_name is not None:
            encoder_name = f"en_{model_name}"
            decoder_name = f"de_{model_name}"

        if encoder_name and decoder_name:
            enc_path = os.path.join(cls.ENCODERS_DIR, encoder_name)
            dec_path = os.path.join(cls.DECODERS_DIR, decoder_name)

        elif lat is not None and inter is not None and dataset is not None:
            enc_path = os.path.join(
                cls.ENCODERS_DIR,
                f"en_int_{inter}_lat_{lat}_{dataset}.keras"
            )
            dec_path = os.path.join(
                cls.DECODERS_DIR,
                f"de_int_{inter}_lat_{lat}_{dataset}.keras"
            )
        else:
            raise ValueError(
                "Invalid combination of arguments. "
                "Use (lat, inter, dataset) or (encoder_name & decoder_name) or model_name."
            )

        # -------- validation --------
        missing = []
        if not os.path.exists(enc_path):
            missing.append(("encoder", enc_path))
        if not os.path.exists(dec_path):
            missing.append(("decoder", dec_path))

        if missing:
            print("\nRequested model not found.\n")

            for kind, path in missing:
                print(f"Missing {kind}: {path}")

            print("\nAvailable encoders:")
            for f in sorted(os.listdir(cls.ENCODERS_DIR)):
                print("  ", f)

            print("\nAvailable decoders:")
            for f in sorted(os.listdir(cls.DECODERS_DIR)):
                print("  ", f)

            raise FileNotFoundError(
                "One or more model files were not found. "
                "See available models above."
            )

        # -------- load models --------

        encoder = load_model(
            enc_path,
            custom_objects=cls._encoder_custom_objects(enc_path)
        )
        decoder = load_model(
            dec_path,
            custom_objects=cls._decoder_custom_objects(dec_path)
        )

        return CVAE(encoder, decoder, original_dim=28 * 28)


    # ---------- predictor ----------

    @classmethod
    def predictor(cls, dataset=None, model_name=None, model_path=None):
        if model_path is not None:
            path = model_path
        elif model_name is not None:
            path = os.path.join(cls.PREDICTORS_DIR, model_name)
        elif dataset is not None:
            if dataset == "early_stop_fashion":
                path = os.path.join(cls.PREDICTORS_DIR, "early_stop_fashion.keras")
            else:
                path = os.path.join(
                    cls.PREDICTORS_DIR, f"CCE_Conv2D_{dataset}.keras"
                )
        else:
            raise ValueError("dataset, model_name or model_path required")

        return load_model(path, custom_objects={"ReshapeLayer": ReshapeLayer})

    # ---------- utilities ----------

    @staticmethod
    def parse_dims_from_key(key):
        parts = key.split("_")
        return int(parts[parts.index("int") + 1]), int(parts[parts.index("lat") + 1])

    @staticmethod
    def _model_pair_key(filename):
        if not filename.endswith(".keras"):
            return None
        if filename.startswith("en_") or filename.startswith("de_"):
            return filename[3:]
        return None

    @classmethod
    def all_cvaes(cls, dataset, lat=None, inter=None):
        encoder_files = os.listdir(cls.ENCODERS_DIR)
        decoder_files = os.listdir(cls.DECODERS_DIR)

        encoders = {
            pair_key: os.path.join(cls.ENCODERS_DIR, f)
            for f in encoder_files
            if (pair_key := cls._model_pair_key(f)) is not None
            and pair_key.endswith(f"_{dataset}.keras")
        }

        decoders = {
            pair_key: os.path.join(cls.DECODERS_DIR, f)
            for f in decoder_files
            if (pair_key := cls._model_pair_key(f)) is not None
            and pair_key.endswith(f"_{dataset}.keras")
        }

        missing_decoders = sorted(set(encoders) - set(decoders))
        missing_encoders = sorted(set(decoders) - set(encoders))
        if missing_decoders or missing_encoders:
            raise FileNotFoundError(
                f"Incomplete CVAE pairs for dataset '{dataset}'. "
                f"Missing decoders for: {missing_decoders}. "
                f"Missing encoders for: {missing_encoders}."
            )

        models = []
        for k in sorted(encoders):
            int_dim, lat_dim = cls.parse_dims_from_key(k)

            if inter and int_dim != inter:
                continue
            if lat and lat_dim != lat:
                continue

            encoder = load_model(
                encoders[k],
                custom_objects=cls._encoder_custom_objects(encoders[k])
            )
            decoder = load_model(
                decoders[k],
                custom_objects=cls._decoder_custom_objects(decoders[k])
            )

            cvae = CVAE(
                encoder,
                decoder,
                original_dim=28 * 28,
                name=f"cvae_{k.removesuffix('.keras')}",
            )
            cvae.compile(optimizer="adam")
            models.append(cvae)

        return models

    # ---------- data ----------

    @classmethod
    def data(cls, dataset):
        return get_mnist_data(dataset=dataset)
