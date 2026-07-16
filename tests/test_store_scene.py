import types

import pytest

from isaac_datagen.store_scene import build_repopulated_store_scene


def test_repopulated_scene_rejects_mutations(tmp_path):
    """Repopulation drives placement from curated sites, so build_repopulated_store_scene
    never calls apply_mutations. A stray mutation spec would be silently dropped — assert
    it fails loud instead. Fires before any Isaac work (no GPU needed)."""
    store_usd = tmp_path / "fake.usd"
    store_usd.write_text("")                       # StoreSceneSpec.__post_init__ asserts the path exists
    runtime = types.SimpleNamespace(scene_builder_args=dict(
        store_usd=str(store_usd),
        product_patterns=["model_*"],
        grasp_frame_policy="FixedFaceGrasp",
        grasp_frame_policy_args={"face": "-Y"},
        site_catalog="some/keep/catalog",
        mutations=[{"name": "RemoveUntrackedProducts"}],
    ))
    with pytest.raises(AssertionError, match="ignores mutations"):
        build_repopulated_store_scene(runtime, [])
