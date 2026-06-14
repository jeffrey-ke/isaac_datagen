"""GraspableObject filter registry: {name, args} specs select filters from this
module (same idiom as posers.py), applied in order. Import-light so runtime_config
can import FilterSpec without pulling in isaacsim.
"""
from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaac_datagen.objects import GraspableObject


@dataclass
class FilterSpec:
    name: str                                 # filter class name in this module
    args: dict = field(default_factory=dict)  # splatted into that class's ctor


def make_filters(specs: list[FilterSpec]) -> list:
    return [getattr(sys.modules[__name__], spec.name)(**spec.args) for spec in specs]


def filter_objects(objects: list[GraspableObject],
                   specs: list[FilterSpec]) -> list[GraspableObject]:
    """Apply each filter in order; raise if the candidate set is ever empty."""
    for f in make_filters(specs):
        if not objects:
            raise ValueError(f"no GraspableObjects left to feed filter {f!r}")
        objects = f(objects)
    return objects


class ShuffleFilter:
    """Deterministic permutation; seed is mandatory."""

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        order = np.random.RandomState(self.seed).permutation(len(objects))
        return [objects[i] for i in order]


class MetaFilter:
    """Cap how many objects matching a meta-field glob survive: walk `objects` in order
    and keep at most `max` whose meta[key] fnmatches the Unix find-style glob `value`
    (e.g. key='name', value='amazon_*'), dropping the overflow among matches. Objects
    that don't match pass through untouched.
    """

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
