import sys

from isaac_datagen.isaac_utils import bounding_half_extents


def mesh_radius(usd_path: str) -> float:
    """Bounding-sphere radius for the object in usd_path: the full diagonal of its
    bounding box. Frame-agnostic by construction -- safe regardless of exactly where
    within the object the render-time pose origin (the grasp frame) actually sits."""
    from pxr import Usd
    import numpy as np

    stage = Usd.Stage.Open(str(usd_path))
    assert stage, f"{usd_path}: pxr could not open this as a USD stage"
    half = bounding_half_extents(stage.GetPseudoRoot())
    # a stage with no mesh geometry gives pxr's empty-range sentinel (~-3.4e38 per
    # axis), not (0, 0, 0) -- catch it here, or a norm() below would silently turn
    # it into an enormous, nonsensical "radius" instead of failing loud.
    assert all(h >= 0 for h in half), \
        f"{usd_path}: empty/invalid bounding box (no mesh geometry?) -- half_extents={half}"
    return float(np.linalg.norm(half)) * 2


def main():
    print(mesh_radius(sys.argv[1]))


if __name__ == "__main__":
    main()
