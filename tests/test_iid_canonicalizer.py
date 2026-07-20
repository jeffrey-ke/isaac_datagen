import pytest
import torch
from torchvision import tv_tensors

from isaac_datagen.isaac_utils import IidCanonicalizer


def mask(*ids):
    return tv_tensors.Mask(torch.tensor([list(ids)], dtype=torch.int32))


def test_first_seen_id_stays_canonical_across_frames():
    c = IidCanonicalizer()
    m, names, occ = c.canonicalize(mask(13), {13: "detergent010"}, {13: 0.1})
    assert m.tolist() == [[13]] and names == {13: "detergent010"} and occ == {13: 0.1}
    m, names, occ = c.canonicalize(mask(14), {14: "detergent010"}, {14: 0.2})  # sibling id alone
    assert m.tolist() == [[13]]                    # pixels moved onto the canonical id
    assert names == {13: "detergent010"}           # per-frame map is 1:1 on canonical ids
    assert occ == {13: 0.2}                        # sibling row fills the gap (no id-13 row this frame)


def test_both_siblings_in_one_frame_collapse():
    c = IidCanonicalizer()
    c.canonicalize(mask(13), {13: "detergent010"}, {13: 0.1})
    m, names, occ = c.canonicalize(
        mask(13, 14), {13: "detergent010", 14: "detergent010"}, {13: 0.3, 14: 0.9})
    assert m.tolist() == [[13, 13]]
    assert names == {13: "detergent010"}
    assert occ == {13: 0.3}                        # canonical id's own row wins over the sibling's


def test_scenery_and_second_object_untouched():
    c = IidCanonicalizer()
    m, names, occ = c.canonicalize(
        mask(13, 40, 7), {13: "detergent010", 7: "cereal001"}, {13: 0.1, 7: 0.2})
    assert m.tolist() == [[13, 40, 7]]             # 40 is unnamed scenery: never remapped
    assert names == {13: "detergent010", 7: "cereal001"}
    assert occ == {13: 0.1, 7: 0.2}


def test_no_remap_frame_returns_inputs_unchanged():
    c = IidCanonicalizer()
    m_in = mask(7)
    m, names, occ = c.canonicalize(m_in, {7: "cereal001"}, {7: 0.5})
    assert m is m_in                               # common case: no copy, no work


def test_id_reuse_for_different_name_raises():
    c = IidCanonicalizer()
    c.canonicalize(mask(13), {13: "detergent010"}, {13: 0.1})
    with pytest.raises(ValueError, match="renamed"):
        c.canonicalize(mask(13), {13: "sauces001"}, {13: 0.1})
