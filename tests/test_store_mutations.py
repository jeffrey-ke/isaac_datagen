import numpy as np
import pytest
import yaml

from isaac_datagen.store_mutations import ProductSite, Site, _orthonormal_rotation, fit_scale, load_sites

KEEP = "assets/optflow_objects/store001-optflow-objects-keep"   # via the assets symlink; cwd = isaac_datagen


def test_load_sites_real_keep_catalog():
    sites = load_sites(KEEP)
    assert len(sites) == 283
    assert all(isinstance(s, Site) for s in sites)
    assert all(s.store_prim.endswith("/v_0") for s in sites)     # store_prim points at the v_0 geometry prim
    assert all(s.grasp.shape == (4, 4) for s in sites)
    assert len({s.cls for s in sites}) == 42


def test_orthonormal_rotation_nonuniform_scale():
    rz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    l2w = np.eye(4)
    l2w[:3, :3] = rz @ np.diag([0.6, 0.5, 0.6])   # real store001 shelf-product scale
    assert np.allclose(_orthonormal_rotation(l2w), rz)


def test_orthonormal_rotation_rejects_shear():
    l2w = np.eye(4)
    l2w[0, 1] = 0.3
    with pytest.raises(AssertionError, match="shear"):
        _orthonormal_rotation(l2w)


def test_load_sites_rejects_catalog_without_store_prim(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir(parents=True)
    (meta / "meta_0000.yaml").write_text(yaml.safe_dump({"name": "x001", "class": "x001"}))
    with pytest.raises(AssertionError, match="store_prim"):
        load_sites(str(tmp_path))


def _site(ext=(0.1, 0.2, 0.3), scale=(1.0, 1.0, 1.0), grasp_rot=np.eye(3)):
    l2w = np.eye(4)
    l2w[:3, :3] = np.diag(scale)
    grasp = np.eye(4)
    grasp[:3, :3] = grasp_rot
    return ProductSite(name="s", path="/W/s", lo=np.zeros(3), hi=np.asarray(ext),
                       l2w=l2w, grasp=grasp)


def test_fit_scale_noop_when_fits():
    assert fit_scale(_site(), np.array([0.1, 0.2, 0.3]), np.eye(4), 1.0) == 1.0
    assert fit_scale(_site(), np.array([0.01, 0.01, 0.01]), np.eye(4), 1.0) == 1.0  # never enlarge


def test_fit_scale_height_binds():
    # ratios (0.5, 0.25, 2.0): the slim-but-tall bottle shrinks by height
    assert fit_scale(_site(), np.array([0.05, 0.05, 0.6]), np.eye(4), 1.0) == pytest.approx(0.5)


def test_fit_scale_girth_binds():
    # ratios (3.0, 0.5, 1/3): worst axis is width
    assert fit_scale(_site(), np.array([0.3, 0.1, 0.1]), np.eye(4), 1.0) == pytest.approx(1 / 3)


def test_fit_scale_yaw_pairs_width_with_depth():
    rz90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    ext_r = np.array([0.3, 0.1, 0.2])
    # unrotated: width 0.3 vs slot 0.1 -> shrink 1/3; the 90-degree grasp alignment
    # turns the object, pairing its 0.3 with the slot's 0.2 axis instead
    assert fit_scale(_site(ext=(0.1, 0.2, 0.3)), ext_r, np.eye(4), 1.0) == pytest.approx(1 / 3)
    assert fit_scale(_site(ext=(0.1, 0.2, 0.3), grasp_rot=rz90), ext_r, np.eye(4), 1.0) \
        == pytest.approx(2 / 3)   # rotated extents (0.1, 0.3, 0.2) vs (0.1, 0.2, 0.3)


def test_fit_scale_honors_nonuniform_site_scale():
    # occupant local ext 0.2^3 under l2w scale (0.5,1,1) -> world slot (0.1, 0.2, 0.2)
    s = _site(ext=(0.2, 0.2, 0.2), scale=(0.5, 1.0, 1.0))
    assert fit_scale(s, np.array([0.1, 0.1, 0.1]), np.eye(4), 1.0) == 1.0
    assert fit_scale(s, np.array([0.2, 0.1, 0.1]), np.eye(4), 1.0) == pytest.approx(0.5)


def test_fit_scale_threshold_scales_allowance():
    assert fit_scale(_site(), np.array([0.1, 0.2, 0.3]), np.eye(4), 0.5) == pytest.approx(0.5)


def test_fit_scale_rejects_tilted_grasp():
    c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
    rx30 = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    with pytest.raises(AssertionError, match="pure yaw"):
        fit_scale(_site(grasp_rot=rx30), np.array([0.1, 0.1, 0.1]), np.eye(4), 1.0)
