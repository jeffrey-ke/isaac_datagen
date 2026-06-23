"""Apples-to-apples cluster-viz comparison of two DIFT-family descriptors on one ObsMask.

Sibling of ``viz_clusters.py`` for the DIFT-vs-CleanDIFT question: same scene, same KMeans seeds, same
backbone+resolution (whatever the two configs share) so the only variable is the descriptor. Reuses
``viz_clusters``' parts wholesale — ``extract_fpn_volumes`` (per-up-block volumes), ``cluster_viz``
(faiss KMeans, ``seed=0``), ``cluster_panel`` (rendering) — and only adds the stacked two-row figure.
``CleanDiftFpn`` is a ``DiftFpn`` subclass, so its volumes come out of ``extract_fpn_volumes`` unchanged.

Seeded clustering (``--seed-iids`` or ``--init``) is what makes the two rows comparable: a cluster is
seeded at the same pixel in both descriptors, so its colour/number/legend mean the same thing across
rows. Each descriptor is loaded, its volumes extracted to CPU numpy, then freed before the next, so
peak VRAM is one model — not both at once.

Usage (from isaac_datagen/, with faiss + a compatible numpy):
    env -u PYTHONPATH uv run --with 'faiss-cpu==1.8.0' --with 'numpy==1.26.0' \
        src/isaac_datagen/viz_clusters_compare.py <render_dir> \
        --dift-config <reference_matching>/configs/fpn_dift_sd21.yaml \
        --cleandift-config <reference_matching>/configs/fpn_cleandift.yaml \
        --idx 0 [--seed-iids | --num-clusters K [--init "x0,y0;..."]] [--device cuda] [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch

from isaac_datagen.viz_clusters import (alpha_crop, cluster_panel, cluster_viz, extract_fpn_volumes,
                                        id_centroid_seeds, parse_init)
from reference_matching import descriptor as descriptor_module
from vision_core.datastructs import ObsMask, ObsMaskDescriptorMetadata
from vision_core.viz import panel_grid, rgba_chw_to_rgb, save_figure


def _clustered_volumes(config_path, om, num_clusters, seeds, device):
    """Build the descriptor from config, extract its per-up-block volumes, free it, then cluster each
    volume with the shared seeds. Returns (label, {key: (H, W) cluster ids}, strides, channels)."""
    descriptor = descriptor_module.from_config(config_path).to(device)
    if not isinstance(descriptor, descriptor_module.DiftFpn):
        raise SystemExit(f"{config_path} builds {type(descriptor).__name__}; the comparison needs an "
                         "FPN descriptor (DiftFpn / CleanDiftFpn) for the per-up-block grid.")
    label = type(descriptor).__name__
    key_to_volume = extract_fpn_volumes(om.obs, descriptor, device)   # {key: (C_k, H, W)} on CPU
    strides, channels = dict(descriptor.strides), dict(descriptor.channels)
    del descriptor                                                    # free SD VRAM before the next model
    torch.cuda.empty_cache()
    key_to_labels = {k: cluster_viz(v, num_clusters, seeds) for k, v in key_to_volume.items()}
    return label, key_to_labels, strides, channels


def cluster_compare_figure(om, rows, init_xy, strides, channels, *, alpha=0.5, cmap="tab20",
                           cluster_labels=None, title=None):
    """One stacked figure: one row per descriptor (obs + per-up-block cluster_panel), aligned columns.

    rows: list of (label, {key: (H, W) cluster ids}); all rows share the same keys, seeds, strides and
    channels, so colours line up cell-for-cell down each column."""
    obs_rgb = rgba_chw_to_rgb(om.obs)
    keys = list(rows[0][1].keys())
    cols = 1 + len(keys)
    fig, axes = panel_grid(len(rows) * cols, cols)
    for r, (label, key_to_labels) in enumerate(rows):
        row_axes = axes[r * cols:(r + 1) * cols]
        row_axes[0].imshow(obs_rgb)
        row_axes[0].set_title(f"{label}  obs", fontsize=8)
        row_axes[0].axis("off")
        for ax, k in zip(row_axes[1:], keys):
            cluster_panel(ax, obs_rgb, key_to_labels[k], init_xy, alpha=alpha, cmap=cmap,
                          cluster_labels=cluster_labels,
                          title=f"{label}  up-block {k}  stride {strides[k]}  C={channels[k]}")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def main():
    p = argparse.ArgumentParser(description="A2A cluster-viz comparison of two DIFT-family FPN descriptors.")
    p.add_argument("render_dir", type=Path)
    p.add_argument("--dift-config", type=Path, required=True, help="baseline descriptor config (e.g. fpn_dift_sd21.yaml)")
    p.add_argument("--cleandift-config", type=Path, required=True, help="descriptor config to compare (e.g. fpn_cleandift.yaml)")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--num-clusters", type=int, default=None, help="required unless --seed-iids")
    p.add_argument("--init", type=str, default=None, help='seed pixel coords "x0,y0;x1,y1;..."')
    p.add_argument("--seed-iids", action="store_true",
                   help="seed one cluster per instance at its iid-mask centroid (aligns colours across rows)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--cmap", type=str, default="tab20")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    assert not (args.init and args.seed_iids), "--init and --seed-iids are mutually exclusive"
    om = ObsMask.deserialize(args.idx, args.render_dir)
    # crop away transparent padding — seeds/figure stay aligned in cropped coords
    om.obs, om.cid_mask, om.iid_mask = alpha_crop(om.obs, om.cid_mask, om.iid_mask)

    cluster_labels = None
    if args.seed_iids:
        iids, init_xy = id_centroid_seeds(om.iid_mask)
        num_clusters = args.num_clusters or len(iids)
        md = ObsMaskDescriptorMetadata.deserialize(0, args.render_dir)
        cluster_labels = {i: md.iid_to_name[iid] for i, iid in enumerate(iids)}
    else:
        init_xy = parse_init(args.init)
        num_clusters = args.num_clusters
    assert num_clusters is not None, "--num-clusters is required without --seed-iids"
    if init_xy is not None:
        assert init_xy.shape[0] == num_clusters, (
            f"{init_xy.shape[0]} seed points but num_clusters={num_clusters}")

    # one model in VRAM at a time; both rows share init_xy so cluster colours align
    rows, geom = [], None
    for config_path in (args.dift_config, args.cleandift_config):
        label, key_to_labels, strides, channels = _clustered_volumes(
            config_path, om, num_clusters, init_xy, args.device)
        rows.append((label, key_to_labels))
        geom = (strides, channels)   # identical across DIFT-family FPNs; last wins

    title = f"{args.render_dir.name}  frame {args.idx:04d}  k={num_clusters}  (seeds aligned across rows)"
    fig = cluster_compare_figure(om, rows, init_xy, *geom, alpha=args.alpha, cmap=args.cmap,
                                 cluster_labels=cluster_labels, title=title)

    out_path = args.out or (args.render_dir.parent / f"{args.render_dir.name}_viz_clusters_cleandift"
                            / f"compare_{args.idx:04d}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, args.dpi)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
