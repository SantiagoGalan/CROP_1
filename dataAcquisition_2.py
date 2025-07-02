import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical


def load_datasets():
    """
    Carga los datasets MNIST y Fashion MNIST.

    Returns:
        x1_train, y1_train: imágenes y etiquetas de entrenamiento de MNIST
        x2_train, y2_train: imágenes y etiquetas de entrenamiento de Fashion MNIST
        x1_test, y1_test: imágenes y etiquetas de prueba de MNIST
        x2_test, y2_test: imágenes y etiquetas de prueba de Fashion MNIST
    """
    (x1_train, y1_train), (x1_test, y1_test) = mnist.load_data()
    (x2_train, y2_train), (x2_test, y2_test) = fashion_mnist.load_data()
    return x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test


def preprocess(x, y, num_classes=10):
    """
    Normaliza y expande dimensiones de imágenes, y convierte etiquetas a one-hot.

    Args:
        x: array de imágenes en escala de grises con forma (N, H, W)
        y: array de etiquetas con forma (N,)
        num_classes: número de clases para one-hot

    Returns:
        x_proc: array de imágenes normalizadas y con shape (N, H, W, 1)
        y_proc: array de etiquetas one-hot con shape (N, num_classes)
    """
    x_proc = x.astype('float32') / 255.0
    x_proc = np.expand_dims(x_proc, axis=-1)
    y_proc = to_categorical(y, num_classes)
    return x_proc, y_proc


def split_train_val(x, y, val_split=0.1, shuffle=True):
    """
    Separa un conjunto de entrenamiento en entrenamiento y validación.

    Args:
        x: array de datos
        y: array de etiquetas
        val_split: fracción de muestras para validación (0 < val_split < 1)
        shuffle: si True, baraja los datos antes de separar

    Returns:
        x_train, y_train, x_val, y_val
    """
    N = x.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    split_at = int(N * (1 - val_split))
    train_idx, val_idx = idx[:split_at], idx[split_at:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def mix_images(x1, x2, mode='average'):
    """
    Genera una mezcla par a par de dos lotes de imágenes.

    Args:
        x1, x2: arrays de imágenes con misma forma (N, H, W, C)
        mode: 'average' para promedio, 'max' para valor máximo píxel a píxel

    Returns:
        x_mix: array mezclado con misma forma que x1/x2
    """
    if mode == 'average':
        return (x1 + x2) / 2.0
    elif mode == 'max':
        return np.maximum(x1, x2)
    else:
        raise ValueError(f"Modo de mezcla desconocido: {mode}")


def get_data(val_split=0.1, mix_mode='average', num_classes=10, seed=None):
    """
    Pipeline completo de adquisición y preprocesamiento de datos.

    Args:
        val_split: fracción para validación
        mix_mode: 'average' o 'max'
        num_classes: número de clases para one-hot
        seed: semilla para reproducibilidad (np.random.seed)

    Returns:
        dict con llaves:
            x1_train, y1_train, x2_train, y2_train,
            x1_val, y1_val, x2_val, y2_val,
            x1_test, y1_test, x2_test, y2_test,
            x_train_mix, x_test_mix
    """
    if seed is not None:
        np.random.seed(seed)

    # Carga cruda
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test = load_datasets()

    # Preprocesamiento
    x1_train, y1_train = preprocess(x1_train, y1_train, num_classes)
    x2_train, y2_train = preprocess(x2_train, y2_train, num_classes)
    x1_test, y1_test   = preprocess(x1_test, y1_test, num_classes)
    x2_test, y2_test   = preprocess(x2_test, y2_test, num_classes)

    # Split train/val
    x1_train, y1_train, x1_val, y1_val = split_train_val(x1_train, y1_train, val_split)
    x2_train, y2_train, x2_val, y2_val = split_train_val(x2_train, y2_train, val_split)

    # Mezclas
    x_train_mix = mix_images(x1_train, x2_train, mode=mix_mode)
    x_test_mix  = mix_images(x1_test, x2_test, mode=mix_mode)

    return {
        'x1_train': x1_train, 'y1_train': y1_train,
        'x2_train': x2_train, 'y2_train': y2_train,
        'x1_val':   x1_val,   'y1_val':   y1_val,
        'x2_val':   x2_val,   'y2_val':   y2_val,
        'x1_test':  x1_test,  'y1_test':  y1_test,
        'x2_test':  x2_test,  'y2_test':  y2_test,
        'x_train_mix': x_train_mix,
        'x_test_mix':  x_test_mix
    }
