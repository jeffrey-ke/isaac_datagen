"""KMeans cluster viz of dense DIFT descriptors on an ObsMask observation.

Two independent parts: ``extract_descriptor_volume`` turns ``ObsMask.obs`` into
a plain (C, H, W) descriptor volume at obs resolution (owning all descriptor
output geometry); ``cluster_viz`` kmeans-clusters the per-pixel descriptors of
any such volume into an (H, W) cluster-id mask. Optional (K, 2) pixel coords
seed the centroids with the descriptors sampled at those locations.

Usage:
    uv run --with scikit-learn src/isaac_datagen/viz_clusters.py <render_dir>
        --idx 0 --num-clusters 8
        [--init "x0,y0;x1,y1;..."] [--descriptor-config PATH] [--device cuda]
        [--out PATH] [--alpha 0.5] [--cmap tab20] [--dpi 200]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from reference_matching import descriptor as descriptor_module
from vision_core.datastructs import ObsMask, ObsMaskMetadata
from vision_core.viz import (OUTSIDE_LEGEND_KW, annotate_points, assign_colors, color_legend,
                             mask_centroid, overlay_id_masks, panel_grid, rgba_chw_to_rgb,
                             save_figure)


def extract_descriptor_volume(observation, descriptor, device="cuda") -> np.ndarray:
    """ObsMask.obs (4, H, W) → DIFT descriptor volume (C, H, W) numpy, resized
    to the observation's dimensions. Owns all descriptor-output geometry: the
    row-major (N, C) un-flatten and the feature-grid → obs-pixel resize."""
    with torch.inference_mode():
        feats = descriptor(observation.unsqueeze(0).to(device))  # (1, N, C)
    feats = feats.squeeze(0).float().cpu()                       # (N, C)
    n, c = feats.shape
    grid = round(n ** 0.5)
    assert grid * grid == n, f"non-square feature map: N={n}"
    volume = feats.T.reshape(c, grid, grid)   # inverts descriptor's .flatten(1).T
    h, w = observation.shape[-2:]
    return F.interpolate(volume[None], size=(h, w), mode="bilinear",
                         align_corners=False).squeeze(0).numpy()


def cluster_viz(volume, num_clusters, cluster_initialization=None):
    """KMeans over the per-pixel descriptors of a (C, H, W) spatial volume
    (numpy or torch) → (H, W) int cluster-id mask.

    cluster_initialization: (num_clusters, 2) pixel (x, y) coords into the
    volume's H, W, or None. When given, the descriptors at those pixels seed
    the centroids; otherwise k-means++.
    """
    if cluster_initialization is not None:
        assert num_clusters == cluster_initialization.shape[0], (
            f"num_clusters={num_clusters} != init points={cluster_initialization.shape[0]}")

    volume = np.asarray(volume)
    c, h, w = volume.shape
    feats = volume.reshape(c, -1).T                              # (H*W, C)
    if cluster_initialization is not None:
        xy = np.asarray(cluster_initialization).round().astype(int)
        init = volume[:, xy[:, 1].clip(0, h - 1), xy[:, 0].clip(0, w - 1)].T  # (K, C)
        km = KMeans(n_clusters=num_clusters, init=init, n_init=1, random_state=0)
    else:
        km = KMeans(n_clusters=num_clusters, init="k-means++", n_init=1, random_state=0)
    return km.fit_predict(feats).reshape(h, w)


def alpha_crop(obs, *masks):
    """Crop an RGBA (4, H, W) obs (and any aligned (H, W) masks) to the
    bounding box of its non-transparent pixels. Returns (obs, *masks)."""
    ys, xs = np.nonzero(np.asarray(obs[3]) > 0)
    assert ys.size, "fully transparent observation"
    rows = slice(ys.min(), ys.max() + 1)
    cols = slice(xs.min(), xs.max() + 1)
    return obs[:, rows, cols], *(m[rows, cols] for m in masks)


def cid_centroid_seeds(cid_mask):
    """(H, W) cid mask → ([cid, ...], (K, 2) (x, y) centroids), one seed per
    non-background class present. A class's union mask can be spatially
    disjoint, so its centroid may land between instances — fine for seeding."""
    cidm = np.asarray(cid_mask)
    cids = [int(c) for c in np.unique(cidm) if c != 0]
    seeds = [(c, mask_centroid(cidm == c)) for c in cids]
    seeds = [(c, xy) for c, xy in seeds if xy is not None]
    return [c for c, _ in seeds], np.array([xy for _, xy in seeds])


def cluster_id_annotations(labels_hw, min_area_frac=0.001):
    """(H, W) cluster ids → [(x, y, "id"), ...], one per connected component of
    each cluster (clusters are disjoint in image space), skipping specks below
    `min_area_frac` of the image."""
    from scipy.ndimage import label as cc_label
    min_area = min_area_frac * labels_hw.size
    points = []
    for i in np.unique(labels_hw):
        components, n = cc_label(labels_hw == i)
        for k in range(1, n + 1):
            comp = components == k
            if comp.sum() >= min_area and (xy := mask_centroid(comp)) is not None:
                points.append((*xy, str(i)))
    return points


def cluster_figure(om, labels_hw, init_xy, *, alpha=0.5, cmap="tab20", title=None,
                   cluster_labels=None):
    """ObsMask + (H, W) cluster ids (+ optional seed coords) → 2-panel Figure:
    raw obs | cluster overlay with legend and seed scatter.

    cluster_labels: optional {cluster id → legend label} (e.g. class names when
    seeding from cid centroids); defaults to "cluster {i}"."""
    obs_rgb = rgba_chw_to_rgb(om.obs)
    ids = sorted(np.unique(labels_hw).tolist())
    id_to_color = assign_colors(ids, cmap)

    fig, (ax_obs, ax_clu) = panel_grid(2, 2)
    ax_obs.imshow(obs_rgb)
    ax_obs.set_title("obs", fontsize=8)
    ax_obs.axis("off")

    ax_clu.imshow(overlay_id_masks(obs_rgb, labels_hw, id_to_color, alpha))
    if init_xy is not None:
        init_xy = np.asarray(init_xy)
        ax_clu.scatter(init_xy[:, 0], init_xy[:, 1], c="white", edgecolors="black",
                       s=40, linewidths=1.0, zorder=3, label="seed")
    annotate_points(ax_clu, cluster_id_annotations(labels_hw))
    cluster_labels = cluster_labels or {}
    color_legend(ax_clu, id_to_color,
                 {i: f"{i}: {cluster_labels.get(i, 'cluster')}" for i in ids},
                 **OUTSIDE_LEGEND_KW)
    ax_clu.set_title(f"{len(ids)} clusters", fontsize=8)
    ax_clu.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def parse_init(spec):
    """'x0,y0;x1,y1;...' → (K, 2) float array, or None."""
    if spec is None:
        return None
    return np.array([[float(v) for v in pt.split(",")] for pt in spec.split(";")])


def main():
    p = argparse.ArgumentParser(description="KMeans cluster viz of DIFT descriptors on ObsMask.obs.")
    p.add_argument("render_dir", type=Path)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--num-clusters", type=int, default=None,
                   help="required unless --seed-cids (which sets it to #classes present)")
    p.add_argument("--init", type=str, default=None, help='seed pixel coords "x0,y0;x1,y1;..."')
    p.add_argument("--seed-cids", action="store_true",
                   help="seed one cluster per class at its cid-mask centroid")
    p.add_argument("--descriptor-config", type=Path, default=None,
                   help="default: <render_dir>/descriptor.yaml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--cmap", type=str, default="tab20")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    assert not (args.init and args.seed_cids), "--init and --seed-cids are mutually exclusive"
    om = ObsMask.deserialize(args.idx, args.render_dir)
    # crop away transparent padding — seeds/figure stay aligned in cropped coords
    om.obs, om.cid_mask = alpha_crop(om.obs, om.cid_mask)

    cluster_labels = None
    if args.seed_cids:
        cids, init_xy = cid_centroid_seeds(om.cid_mask)
        num_clusters = args.num_clusters or len(cids)
        md = ObsMaskMetadata.deserialize(0, args.render_dir)
        cluster_labels = {i: md.cid_to_class[c] for i, c in enumerate(cids)}
    else:
        init_xy = parse_init(args.init)
        num_clusters = args.num_clusters
    assert num_clusters is not None, "--num-clusters is required without --seed-cids"
    if init_xy is not None:
        assert init_xy.shape[0] == num_clusters, (
            f"{init_xy.shape[0]} seed points but num_clusters={num_clusters}")

    config_path = args.descriptor_config or args.render_dir / "descriptor.yaml"
    descriptor = descriptor_module.from_config(config_path).to(args.device)
    volume = extract_descriptor_volume(om.obs, descriptor, args.device)  # (C, H, W)
    del descriptor   # free SD-1.5 VRAM before clustering/matplotlib work
    labels_hw = cluster_viz(volume, num_clusters, init_xy)

    fig = cluster_figure(om, labels_hw, init_xy, alpha=args.alpha, cmap=args.cmap,
                         cluster_labels=cluster_labels,
                         title=f"{args.render_dir.name}  frame {args.idx:04d}  k={num_clusters}")
    out_path = args.out or (args.render_dir.parent / f"{args.render_dir.name}_viz_clusters"
                            / f"clusters_{args.idx:04d}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, args.dpi)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
