"""GraspableObject filter registry: {name, args} specs select filters from this
module (same idiom as posers.py), applied in order. Import-light so runtime_config
can import FilterSpec without pulling in isaacsim.
"""
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
    name: str                                 # filter class name in this module
    args: dict = field(default_factory=dict)  # splatted into that class's ctor


def make_filters(specs: list[FilterSpec]) -> list:
    return [getattr(sys.modules[__name__], spec.name)(**spec.args) for spec in specs]


def filter_objects(objects: list[GraspableObject],
                   specs: list[FilterSpec]) -> list[GraspableObject]:
    """Apply each filter in order; raise if the candidate set is ever empty.

    The collect_* loaders guard name-uniqueness on load, but filters run after them and
    may mint new objects (ReplicateFilter), so re-assert globally-unique meta["name"] on
    the final set: it is the placed prim-path component (add_object) and the writer catalog
    join key, both silently last-wins on a clash."""
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
    """Deterministic permutation; seed is mandatory."""

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


class RegexFilter:
    """Keep only objects whose meta[key] matches the regex `value` (via re.search),
    dropping every non-match. Unlike MetaFilter (a glob quota that passes non-matches
    through), this is an inclusion filter and supports alternation, e.g.
    key='class', value='cheezit|mustard|amazon_.*'.
    """

    def __init__(self, key: str, value: str):
        self.key = key
        self.pattern = re.compile(value)

    def __call__(self, objects: list[GraspableObject]) -> list[GraspableObject]:
        return [o for o in objects
                if self.pattern.search(str(o.meta[self.key]))]
