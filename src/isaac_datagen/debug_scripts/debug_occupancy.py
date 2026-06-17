"""Show the OccupancyGrid state + which boxes are graspable (no Isaac Sim needed).

Reconstructs exactly what create_stack_of_objects() does for the
reference_segmentation run, then prints:
  - a side-view of the grid (real object / phantom-occupied / graspable),
  - a per-placed-box table of is_front / is_top / is_graspable,
  - the rule definitions and why only one box qualifies.

The graspability logic is pure numpy, so this runs instantly without booting
the sim. Run from src/isaac_datagen/:

    uv run debug_occupancy.py configs/randomized.yaml
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import yaml

from isaac_datagen.objects import OccupancyGrid


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    dims = tuple(cfg["pallet_dims"])
    gpath = cfg["objects_path"]

    # Mirror clean_datagen.reference_segmentation: objects = collect_objects(...)
    metas = sorted(glob.glob(os.path.join(gpath, "meta", "meta_*.yaml")))
    names = [yaml.safe_load(open(m)).get("name", "?") for m in metas]

    # Full-wall policy: the stack takes the first `capacity` objects.
    capacity = int(np.prod(dims))
    sliced_names = names[:capacity]
    n = len(sliced_names)
    # bbox dims are irrelevant to graspability (they only set slot translation).
    grid = OccupancyGrid(dims, (1.0, 1.0, 1.0))
    seq = grid.sequence                      # nonzero order: k fastest, then j, then i
    placed = seq[:min(n, capacity)]          # objects fill the FIRST n sequence slots
    placed_coord_to_name = dict(zip(placed, sliced_names))
    graspable = [c for c in placed if grid.is_front(*c) and grid.is_top(*c)]
    graspable_set = set(graspable)
    placed_set = set(placed)

    gx, gy, gz = dims
    print(f"pallet_dims = {dims}  (capacity = {capacity})")
    print(f"objects placed = {n}  (objects[4:11]); grid is initialized FULL: np.ones({dims})")
    print(f"=> grid claims all {capacity} slots occupied, but only {n} have a real box.\n")

    # Side view per depth layer j: rows k (top->bottom), cols i (0..gx-1).
    legend = "  G = graspable real box   # = real box (not graspable)   . = phantom (grid=1, no box)"
    for j in range(gy):
        print(f"side view (depth j={j}):   columns i=0..{gx-1} left->right, rows k top->bottom")
        header = "        " + " ".join(f"{i:>2}" for i in range(gx))
        print(header)
        for k in range(gz - 1, -1, -1):
            cells = []
            for i in range(gx):
                c = (i, j, k)
                if c in graspable_set:
                    cells.append(" G")
                elif c in placed_set:
                    cells.append(" #")
                else:
                    cells.append(" .")
            print(f"  k={k}  " + " ".join(cells))
        print(legend + "\n")

    # Per-placed-box graspability table.
    print("placed boxes (in placement order = sequence order):")
    print(f"  {'coord (i,j,k)':<14} {'name':<12} {'is_front':<9} {'is_top':<7} graspable")
    for c in placed:
        f = grid.is_front(*c)
        t = grid.is_top(*c)
        nm = placed_coord_to_name.get(c, "?")
        flag = "  <== GRASP" if (f and t) else ""
        print(f"  {str(c):<14} {nm:<12} {str(f):<9} {str(t):<7} {f and t}{flag}")

    print(f"\n=> {len(graspable)} graspable box(es): "
          f"{[(c, placed_coord_to_name[c]) for c in graspable]}")

    full = (n == capacity)
    print(
        "\nRules (objects.py):\n"
        "  is_front(i,j,k): j == 0  OR  no occupied cell at any y<j in column (i,*,k)\n"
        "  is_top(i,j,k)  : k == K-1 OR  no occupied cell at any z>k in column (i,j,*)\n"
        "  graspable      : is_front AND is_top\n"
        f"\nStatic full-wall policy (grid = all-ones, every slot filled): {'OK' if full else 'NOT FULL'}\n"
        f"  - {n}/{capacity} slots have a real box.\n"
        "  - Pallet is 1 deep (j only 0), so is_front is ALWAYS true.\n"
        f"  - With a full wall, is_top is true only at the top layer k={gz-1}, so the\n"
        f"    graspable boxes are exactly the top row ({gx * gy} of them).\n"
        + ("" if full else
           "  !! UNDER-FILLED: lower-than-top boxes that are physically exposed look\n"
           "     'buried' under phantom slots. build_scene now raises in this case.")
    )


if __name__ == "__main__":
    main()
