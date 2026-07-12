from __future__ import annotations

import dataclasses
import fnmatch
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaac_datagen.objects import GraspableObject


@dataclass
class FilterSpec:
    name: str
    args: dict = field(default_factory=dict)


def make_filters(specs: list[FilterSpec]) -> list:
    return [getattr(sys.modules[__name__], spec.name)(**spec.args) for spec in specs]


def filter_objects(objects: list[GraspableObject],
                   specs: list[FilterSpec]) -> list[GraspableObject]:
    for f in make_filters(specs):
        if not objects:
            raise ValueError(f"no GraspableObjects left to feed filter {f!r}")
        objects = f(objects)

    names = [o.meta["name"] for o in objects]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate object names after filtering: {dupes}")
    return objects


class ShuffleFilter:

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        order = np.random.RandomState(self.seed).permutation(len(objects))
        return [objects[i] for i in order]


class ReplicateFilter:
    def __init__(self, count: int, key: str = "name", value: str = "*"):
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        self.count = count
        self.key = key
        self.value = value

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        out = []
        for o in objects:
            if not fnmatch.fnmatchcase(str(o.meta[self.key]), self.value):
                out.append(o)
                continue
            for k in range(self.count):
                name = o.meta["name"] if k == 0 else f"{o.meta['name']}_dup{k}"
                out.append(dataclasses.replace(o, meta={**o.meta, "name": name}))
        return out


class MetaFilter:

    def __init__(self, key: str, value: str, max: int):
        self.key = key
        self.value = value
        self.max = max

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        kept = 0
        out = []
        for o in objects:
            if fnmatch.fnmatchcase(str(o.meta[self.key]), self.value):
                if kept >= self.max:
                    continue
                kept += 1
            out.append(o)
        return out


class RegexFilter:

    def __init__(self, key: str, value: str):
        self.key = key
        self.pattern = re.compile(value)

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        return [o for o in objects
                if self.pattern.search(str(o.meta[self.key]))]
