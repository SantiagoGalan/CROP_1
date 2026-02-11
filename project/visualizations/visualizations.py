import numpy as np
import matplotlib.pyplot as plt


def condiciones(cvae, x_input):
    """
    Muestra cómo se reconstruye una imagen de entrada
    bajo las 10 condiciones posibles (0 a 9),
    incluyendo la imagen original a la izquierda.
    """

    # Asegurar batch dimension
    x_input = np.expand_dims(x_input, axis=0)

    # Repetir la imagen 10 veces
    x_repeated = np.repeat(x_input, repeats=10, axis=0)

    # Condiciones one-hot
    condiciones = np.eye(10)

    # Codificar y reconstruir
    z_mean, z_log_var, z = cvae.encoder.predict([x_repeated, condiciones])
    reconstrucciones = cvae.decoder.predict([z, condiciones])

    # Mostrar
    plt.figure(figsize=(18, 2.5))

    # 🔹 Imagen original
    plt.subplot(1, 11, 1)
    plt.imshow(x_input[0].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # 🔹 Reconstrucciones
    for i in range(10):
        plt.subplot(1, 11, i + 2)
        plt.imshow(reconstrucciones[i].reshape(28, 28), cmap="gray")
        plt.title(f"Clase {i}")
        plt.axis("off")

    plt.suptitle("Reconstrucciones bajo distintas condiciones", y=1.05)
    plt.tight_layout()
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
    latent_dim = cvae.decoder.input_shape[0][1]  # obtiene la dimensión latente del input
    z = np.random.normal(
        size=(num_variantes, latent_dim)
    ) 

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

def latent_space_umap(
    cvae,
    dataset,
    n_points=2000,
    save_path=None,
    title="",
    label_names=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import umap
    from matplotlib.colors import ListedColormap, BoundaryNorm

    z_all = []
    y_all = []
    count = 0

    # -----------------------------
    # Obtención de representaciones latentes
    # -----------------------------
    for (batch, labels), _ in dataset:
        batch = np.array(batch)
        labels = np.array(labels)

        if batch.ndim == 1:
            batch = np.expand_dims(batch, axis=0)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)

        # Encoder condicional
        z_mean, _, z = cvae.encoder.predict([batch, labels], verbose=0)

        z_all.append(z)
        y_all.append(labels)

        count += len(batch)
        if count >= n_points:
            break

    z_all = np.concatenate(z_all, axis=0)[:n_points]
    y_all = np.argmax(np.concatenate(y_all, axis=0)[:n_points], axis=1)

    n_classes = len(label_names)

    # -----------------------------
    # UMAP
    # -----------------------------
    reducer = umap.UMAP(n_components=2, random_state=42)
    z_umap = reducer.fit_transform(z_all)

    # -----------------------------
    # Colormap discreto con labels centrados
    # -----------------------------
    cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])
    boundaries = np.arange(-0.5, n_classes, 1)
    #boundaries = np.arange(-0.5, n_classes + 0.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        z_umap[:, 0],
        z_umap[:, 1],
        c=y_all,
        cmap=cmap,
        norm=norm,
        s=6,
        alpha=0.6
    )

    cbar = plt.colorbar(sc, ticks=np.arange(n_classes))
    cbar.set_ticklabels(label_names)
    cbar.set_label("Etiqueta")

    plt.xlabel("UMAP [0]")
    plt.ylabel("UMAP [1]")
    plt.title(f"Espacio latente con UMAP {title}")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()



def variantes_punto_fijo(cvae, z_fixed=None, num_puntos=5):
    """
    Muestra cómo diferentes puntos latentes generan dígitos bajo distintas condiciones.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generar o usar puntos latentes fijos
    latent_dim = cvae.decoder.input_shape[0][1]
    if z_fixed is None:
        z_fixed = np.random.normal(size=(num_puntos, latent_dim))
    else:
        num_puntos = z_fixed.shape[0]
    
    fig, axes = plt.subplots(num_puntos, 10, figsize=(20, 2*num_puntos))
    axes = np.atleast_2d(axes)
    fig.suptitle('Diferentes dígitos generados desde puntos latentes fijos', fontsize=16)

    for i in range(num_puntos):
        z = np.tile(z_fixed[i:i+1], (10, 1))
        conditions = np.eye(10)

        imgs_generadas = cvae.decoder.predict([z, conditions], verbose=0)

        for j in range(10):
            ax = axes[i, j]
            ax.imshow(imgs_generadas[j].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Dígito {j}')

        axes[i, 0].set_ylabel(f'Punto {i+1}')
    
    plt.tight_layout()
    plt.show()
    
    return z_fixed  # Retornar los puntos para poder reusarlos
