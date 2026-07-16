import sys

from isaac_datagen.scene import build_scene
from isaac_datagen.store_scene import build_store_scene, build_repopulated_store_scene


def get(name: str):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        raise KeyError(name) from e
