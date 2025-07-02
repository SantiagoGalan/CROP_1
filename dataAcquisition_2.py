import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical

def load_datasets():
    (x1_train, y1_train), (x1_test, y1_test) = mnist.load_data()
    (x2_train, y2_train), (x2_test, y2_test) = fashion_mnist.load_data()
    return x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test

def preprocess(x, y, num_classes=10, flatten=False):
    x_proc = x.astype('float32') / 255.0
    if not flatten:
        x_proc = np.expand_dims(x_proc, axis=-1)
    else:
        x_proc = x_proc.reshape((x.shape[0], -1))
    y_proc = to_categorical(y, num_classes)
    return x_proc, y_proc

def split_train_val(x, y, val_split=0.0, shuffle=True):
    N = x.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    split_at = int(N * (1 - val_split))
    train_idx, val_idx = idx[:split_at], idx[split_at:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

def mix_images(x1, x2, mode='average'):
    if mode == 'average':
        return (x1 + x2) / 2.0
    elif mode == 'max':
        return np.maximum(x1, x2)
    else:
        raise ValueError(f"Modo de mezcla desconocido: {mode}")

def get_data(val_split=0.0, mix_mode='average', num_classes=10, seed=None, flatten=False):
    if seed is not None:
        np.random.seed(seed)

    # Carga cruda
    x1_train, y1_train, x2_train, y2_train, x1_test, y1_test, x2_test, y2_test = load_datasets()

    # Preprocesamiento
    x1_train, y1_train = preprocess(x1_train, y1_train, num_classes, flatten)
    x2_train, y2_train = preprocess(x2_train, y2_train, num_classes, flatten)
    x1_test, y1_test   = preprocess(x1_test, y1_test, num_classes, flatten)
    x2_test, y2_test   = preprocess(x2_test, y2_test, num_classes, flatten)

    # Split train/val (sin shuffle para coincidencia determinista)
    x1_train, y1_train, x1_val, y1_val = split_train_val(x1_train, y1_train, val_split, shuffle=False)
    x2_train, y2_train, x2_val, y2_val = split_train_val(x2_train, y2_train, val_split, shuffle=False)

    # Mezcla reproducible
    perm_train = np.random.permutation(x1_train.shape[0])
    perm_test  = np.random.permutation(x1_test.shape[0])

    x1_train_1 = x1_train[perm_train]
    x2_train_1 = x2_train[perm_train]
    x1_test_1  = x1_test[perm_test]
    x2_test_1  = x2_test[perm_test]

    x_train_mix = mix_images(x1_train, x2_train_1, mode=mix_mode)
    x_test_mix  = mix_images(x1_test,  x2_test_1,  mode=mix_mode)

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
