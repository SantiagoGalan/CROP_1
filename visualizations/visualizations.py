import numpy as np
import matplotlib.pyplot as plt


def condiciones(cvae, x_input):
    """
    Muestra cómo se reconstruye una imagen de entrada bajo las 10 condiciones posibles (0 a 9).
    """

    # Asegurar que x_input tenga batch dimension
    # if x_input.shape == (28, 28):
    x_input = np.expand_dims(x_input, axis=0)

    # Repetir la imagen 10 veces
    x_repeated = np.repeat(x_input, repeats=10, axis=0)

    # Crear las 10 condiciones one-hot
    condiciones = np.eye(10)  # (10, 10)

    # Codificar
    z_mean, z_log_var, z = cvae.encoder.predict([x_repeated, condiciones])

    # Reconstruir
    reconstrucciones = cvae.decoder.predict([z, condiciones])

    # Mostrar
    plt.figure(figsize=(15, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(reconstrucciones[i].reshape(28, 28), cmap="gray")
        plt.title(f"Clase {i}")
        plt.axis("off")
    plt.suptitle("Reconstrucciones bajo distintas condiciones")
    plt.show()


def variantes(cvae, condicion_id, num_variantes=10, custom_condition=None):
    """
    Muestra múltiples imágenes generadas para una misma condición.

    Args:
        cvae: modelo CVAE entrenado.
        condicion_id: entero de 0 a 9, la clase condicional deseada.
        num_variantes: número de muestras a generar.
    """

    if custom_condition is not None:
        condiciones = custom_condition
    else:
        condicion = np.eye(10)[condicion_id]
        condiciones = np.repeat([condicion], num_variantes, axis=0)

    # Generar z aleatorios ~ N(0,1)
    latent_dim = cvae.decoder.input_shape[0][
        1
    ]  # obtiene la dimensión latente del input
    z = np.random.normal(
        size=(num_variantes, latent_dim)
    )  # (num_variantes, latent_dim)

    # Generar imágenes con el decoder
    imgs_generadas = cvae.decoder.predict([z, condiciones])

    # Mostrar
    plt.figure(figsize=(15, 2))
    for i in range(num_variantes):
        plt.subplot(1, num_variantes, i + 1)
        plt.imshow(imgs_generadas[i].reshape(28, 28), cmap="gray")
        # plt.imshow(imgs_generadas[i], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Variantes generadas para la clase {condicion_id}")
    plt.show()
    return z


def lattent_space(cvae, dataset):
    import matplotlib.pyplot as plt
    import numpy as np

    z_all = []
    y_all = []

    for (batch, labels), _ in dataset:
        # Obtené z y labels
        z_mean, _, z = cvae.encoder.predict([batch, labels], verbose=0)
        # Concatená z y labels (eje 1)
        z_input = np.concatenate([z, labels], axis=1)
        z_all.append(z_input)
        y_all.append(labels)

    z_all = np.concatenate(z_all, axis=0)
    y_all = np.argmax(np.concatenate(y_all, axis=0), axis=1)

    # Visualizá solo las dos primeras dimensiones de la entrada al decoder
    plt.figure(figsize=(8, 6))
    plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap="tab10", alpha=0.5, s=5)
    plt.colorbar(label="Etiqueta")
    plt.xlabel("z+label [0]")
    plt.ylabel("z+label [1]")
    plt.title("Espacio latente (entrada real al decoder)")
    plt.show()


def latent_space_tsne(cvae, dataset, max_samples=10000, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    z_all = []
    y_all = []
    count = 0
    for (batch, labels), _ in dataset:
        batch = np.array(batch)
        labels = np.array(labels)
        if batch.ndim == 1:
            batch = np.expand_dims(batch, axis=0)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)
        z_mean, _, z = cvae.encoder.predict([batch, labels], verbose=0)
        z_input = np.concatenate([z, labels], axis=1)
        z_all.append(z_input)
        y_all.append(labels)
        count += len(batch)
        # print(count)
        if count >= max_samples:
            break

    z_all = np.concatenate(z_all, axis=0)
    y_all = np.argmax(np.concatenate(y_all, axis=0), axis=1)

    # t-SNE para reducir a 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(z_all)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_all, cmap="tab10", alpha=0.5, s=5)
    plt.colorbar(label="Etiqueta")
    plt.xlabel("t-SNE [0]")
    plt.ylabel("t-SNE [1]")
    plt.title("Espacio latente con t-SNE")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def latent_space_umap(cvae, dataset, max_samples=2000, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import umap

    z_all = []
    y_all = []
    count = 0

    for (batch, labels), _ in dataset:
        batch = np.array(batch)
        labels = np.array(labels)
        if batch.ndim == 1:
            batch = np.expand_dims(batch, axis=0)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)
        z_mean, _, z = cvae.encoder.predict([batch, labels], verbose=0)
        z_input = np.concatenate([z, labels], axis=1)
        z_all.append(z_input)
        y_all.append(labels)
        count += len(batch)
        if count >= max_samples:
            break

    z_all = np.concatenate(z_all, axis=0)[:max_samples]
    y_all = np.argmax(np.concatenate(y_all, axis=0)[:max_samples], axis=1)

    reducer = umap.UMAP(n_components=2, random_state=42)
    z_umap = reducer.fit_transform(z_all)

    # Guardar los datos para análisis externo
    np.save("z_all.npy", z_all)
    np.save("y_all.npy", y_all)

    # (Opcional) también podés guardar el embedding UMAP si querés
    np.save("z_umap.npy", z_umap)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=y_all, cmap="tab10", alpha=0.5, s=5)
    plt.colorbar(label="Etiqueta")
    plt.xlabel("UMAP [0]")
    plt.ylabel("UMAP [1]")
    plt.title("Espacio latente con UMAP")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
