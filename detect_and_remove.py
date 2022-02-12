# Importamos las dependencias del script.
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from imutils import paths


def dhash(image, hash_size=8):
    """
    Calcula el dhash de la imagen de entrada.
    :param image: Imagen a la cuaal le calcularemos el dhash.
    :param hash_size: Número de bytes en el hash resultante.
    """
    # Resdimensionamos la imagen con base al tamaño del hash.
    resized = cv2.resize(image, (hash_size + 1, hash_size))

    # Generamos la imagen de diferencias de píxeles adyacentes.
    diff = resized[:, 1:] > resized[:, :-1]

    # Calculamos el hash.
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])


# Definimos los argumentos del programa.
argument_parser = ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Ruta al dataset.')
argument_parser.add_argument('-r', '--remove', type=int, default=-1,
                             help='Indica si los duplicados deberían removerse o no (cualquier número mayor a 0 significa sí)')
arguments = vars(argument_parser.parse_args())

print('Calculando los hashes...')
# Cargamos la lista de imágenes a procesar.
image_paths = list(paths.list_images(arguments['dataset']))
hashes = {}  # Aquí guardaremos los hashes.

# Computamos el hash de cada imagen. Ten en cuenta que asociamos cada hash a una lista de rutas, por lo que
# si hay imágenes repetidas, pasará que un hash estará vinculado a una lista de más de un elemento.
for image_path in image_paths:
    image = cv2.imread(image_path)
    hash = dhash(image)

    p = hashes.get(hash, [])
    p.append(image_path)
    hashes[hash] = p

# Iteramos sobre los hashes.
for hash, hashed_paths in hashes.items():
    # Si el hash en cuestión está asociado a más de una ruta, entonces tenemos una imagen repetida.
    if len(hashed_paths) > 1:
        # Si no vamos a remover los duplicados, entonces construiremos un montaje para mostrarlos.
        if arguments['remove'] < 1:
            montage = None

            # Construimos un mosaico de todos los duplicados.
            for p in hashed_paths:
                # Leemos y redimensionamos la imagen.
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))

                if montage is None:
                    montage = image
                else:
                    montage = np.hstack([montage, image])

            # Imprimimos información sobre los duplicados y mostramos el montaje en pantalla.
            print(f'Hash: {hash}')
            print(f'# duplicados: {len(hashed_paths)}')
            print('---')
            cv2.imshow('Montaje', montage)
            cv2.waitKey(0)
        else:
            # Removemos todos los duplicados salvo el primero.
            for p in hashed_paths[1:]:
                os.remove(p)
