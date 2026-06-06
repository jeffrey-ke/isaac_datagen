"""KMeans cluster viz of dense DIFT descriptors on an ObsMask observation.

Independent parts: ``extract_descriptor_volume`` (DiftDescriptor) and
``extract_fpn_volumes`` (DiftFpn, one volume per up-block) turn ``ObsMask.obs``
into plain (C, H, W) descriptor volumes at obs resolution (owning all
descriptor output geometry); ``cluster_viz`` faiss-kmeans-clusters the
per-pixel descriptors of any such volume into an (H, W) cluster-id mask.
Optional (K, 2) pixel coords seed the centroids with the descriptors sampled
at those locations; when seeds are given, both the seeded and the unseeded
clustering are emitted for comparison.

Usage:
    uv run --with 'faiss-cpu==1.8.0' --with 'numpy==1.26.0' \
        src/isaac_datagen/viz_clusters.py <render_dir>
        --idx 0 [--num-clusters 8 | --seed-iids]
        [--init "x0,y0;x1,y1;..."] [--descriptor-config PATH] [--device cuda]
        [--out PATH] [--alpha 0.5] [--cmap tab20] [--dpi 200]

    --descriptor-config defaults to <render_dir>/descriptor.yaml; point it at
    reference_matching's configs/fpn_dift.yaml for the per-up-block figure.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from reference_matching import descriptor as descriptor_module
from vision_core.datastructs import ObsMask, ObsMaskMetadata
from vision_core.viz import (OUTSIDE_LEGEND_KW, assign_colors, color_legend, mask_centroid,
                             overlay_id_masks, panel_grid, rgba_chw_to_rgb, save_figure)


def extract_descriptor_volume(observation, descriptor, device="cuda") -> np.ndarray:
    """ObsMask.obs (4, H, W) → DIFT descriptor volume (C, H, W) numpy, resized
    to the observation's dimensions. The descriptor emits spatial features;
    this owns only the feature-grid → obs-pixel resize."""
    with torch.inference_mode():
        feats = descriptor(observation.unsqueeze(0).to(device))  # (1, C, h, w)
    volume = feats.squeeze(0).float().cpu()                      # (C, h, w)
    h, w = observation.shape[-2:]
    return F.interpolate(volume[None], size=(h, w), mode="bilinear",
                         align_corners=False).squeeze(0).numpy()


def extract_fpn_volumes(observation, fpn, device="cuda") -> dict:
    """ObsMask.obs (4, H, W) → {key: (C_k, H, W) numpy}, one volume per FPN key,
    each bilinear-resized to the observation's dimensions. Sibling of
    extract_descriptor_volume for the keyed FPN wrappers (``fpn(fpn.prep(image))``
    call pattern); DiftFpn volumes arrive spatial — no un-flatten."""
    with torch.inference_mode():
        volumes = fpn(fpn.prep(observation).unsqueeze(0).to(device))  # [(1, C_k, h_k, w_k), ...]
    h, w = observation.shape[-2:]
    return {k: F.interpolate(v.float().cpu(), size=(h, w), mode="bilinear",
                             align_corners=False).squeeze(0).numpy()
            for k, v in zip(fpn.keys, volumes, strict=True)}


def cluster_viz(volume, num_clusters, cluster_initialization=None):
    """faiss KMeans over the per-pixel descriptors of a (C, H, W) spatial
    volume (numpy or torch) → (H, W) int cluster-id mask.

    cluster_initialization: (num_clusters, 2) pixel (x, y) coords into the
    volume's H, W, or None. When given, the descriptors at those pixels seed
    the centroids; otherwise faiss's default (random-sample) init.
    """
    if cluster_initialization is not None:
        assert num_clusters == cluster_initialization.shape[0], (
            f"num_clusters={num_clusters} != init points={cluster_initialization.shape[0]}")

    volume = np.asarray(volume)
    c, h, w = volume.shape
    feats = np.ascontiguousarray(volume.reshape(c, -1).T, dtype=np.float32)  # (H*W, C)
    init = None
    if cluster_initialization is not None:
        xy = np.asarray(cluster_initialization).round().astype(int)
        init = np.ascontiguousarray(                                          # (K, C)
            volume[:, xy[:, 1].clip(0, h - 1), xy[:, 0].clip(0, w - 1)].T, dtype=np.float32)
    km = faiss.Kmeans(c, num_clusters, niter=25, seed=0)
    km.train(feats, init_centroids=init)
    _, labels = km.index.search(feats, 1)   # assign every pixel to its nearest centroid
    return labels.ravel().reshape(h, w)


def alpha_crop(obs, *masks):
    """Crop an RGBA (4, H, W) obs (and any aligned (H, W) masks) to the
    bounding box of its non-transparent pixels. Returns (obs, *masks)."""
    ys, xs = np.nonzero(np.asarray(obs[3]) > 0)
    assert ys.size, "fully transparent observation"
    rows = slice(ys.min(), ys.max() + 1)
    cols = slice(xs.min(), xs.max() + 1)
    return obs[:, rows, cols], *(m[rows, cols] for m in masks)


def id_centroid_seeds(id_mask):
    """(H, W) id mask (iid or cid space) → ([id, ...], (K, 2) (x, y) centroids),
    one seed per non-background id present, at that id's mask centroid."""
    idm = np.asarray(id_mask)
    ids = [int(i) for i in np.unique(idm) if i != 0]
    seeds = [(i, mask_centroid(idm == i)) for i in ids]
    seeds = [(i, xy) for i, xy in seeds if xy is not None]
    return [i for i, _ in seeds], np.array([xy for _, xy in seeds])


def cluster_panel(ax, obs_rgb, labels_hw, init_xy, *, alpha=0.5, cmap="tab20",
                  cluster_labels=None, title=None):
    """One axis: cluster overlay + seed scatter + legend.

    cluster_labels: optional {cluster id → legend label} (e.g. instance names
    when seeding from iid centroids); defaults to "cluster"."""
    ids = sorted(np.unique(labels_hw).tolist())
    id_to_color = assign_colors(ids, cmap)
    ax.imshow(overlay_id_masks(obs_rgb, labels_hw, id_to_color, alpha))
    if init_xy is not None:
        init_xy = np.asarray(init_xy)
        ax.scatter(init_xy[:, 0], init_xy[:, 1], c="white", edgecolors="black",
                   s=40, linewidths=1.0, zorder=3, label="seed")
    cluster_labels = cluster_labels or {}
    color_legend(ax, id_to_color,
                 {i: f"{i}: {cluster_labels.get(i, 'cluster')}" for i in ids},
                 **OUTSIDE_LEGEND_KW)
    ax.set_title(title or f"{len(ids)} clusters", fontsize=8)
    ax.axis("off")


def cluster_figure(om, labels_hw, init_xy, *, alpha=0.5, cmap="tab20", title=None,
                   cluster_labels=None):
    """ObsMask + (H, W) cluster ids (+ optional seed coords) → 2-panel Figure:
    raw obs | cluster_panel."""
    obs_rgb = rgba_chw_to_rgb(om.obs)
    fig, (ax_obs, ax_clu) = panel_grid(2, 2)
    ax_obs.imshow(obs_rgb)
    ax_obs.set_title("obs", fontsize=8)
    ax_obs.axis("off")
    cluster_panel(ax_clu, obs_rgb, labels_hw, init_xy, alpha=alpha, cmap=cmap,
                  cluster_labels=cluster_labels)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def fpn_cluster_figure(om, key_to_labels, init_xy, strides, channels, *,
                       alpha=0.5, cmap="tab20", cluster_labels=None, title=None, cols=3):
    """ObsMask + per-key (H, W) cluster masks → Figure: obs panel + one
    cluster_panel per FPN level, titled with that level's geometry.

    Each level is clustered independently, so unseeded cluster ids are arbitrary
    per panel; with seeds, cluster i is seeded at the same pixel in every level,
    so colors/numbers/legend align across panels."""
    obs_rgb = rgba_chw_to_rgb(om.obs)
    fig, axes = panel_grid(1 + len(key_to_labels), cols)
    axes[0].imshow(obs_rgb)
    axes[0].set_title("obs", fontsize=8)
    axes[0].axis("off")
    for ax, (k, labels_hw) in zip(axes[1:], key_to_labels.items()):
        cluster_panel(ax, obs_rgb, labels_hw, init_xy, alpha=alpha, cmap=cmap,
                      cluster_labels=cluster_labels,
                      title=f"up-block {k}  stride {strides[k]}  C={channels[k]}")
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
    p.add_argument("--seed-iids", action="store_true",
                   help="seed one cluster per instance at its iid-mask centroid")
    p.add_argument("--descriptor-config", type=Path, default=None,
                   help="default: <render_dir>/descriptor.yaml")
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
        md = ObsMaskMetadata.deserialize(0, args.render_dir)
        cluster_labels = {i: md.iid_to_name[iid] for i, iid in enumerate(iids)}
    else:
        init_xy = parse_init(args.init)
        num_clusters = args.num_clusters
    assert num_clusters is not None, "--num-clusters is required without --seed-iids"
    if init_xy is not None:
        assert init_xy.shape[0] == num_clusters, (
            f"{init_xy.shape[0]} seed points but num_clusters={num_clusters}")

    # with seeds, emit BOTH the seeded and the unseeded clustering for comparison
    runs = ([("", None, None)] if init_xy is None
            else [("_seeded", init_xy, cluster_labels), ("_unseeded", None, None)])

    config_path = args.descriptor_config or args.render_dir / "descriptor.yaml"
    descriptor = descriptor_module.from_config(config_path).to(args.device)
    title_base = f"{args.render_dir.name}  frame {args.idx:04d}  k={num_clusters}"
    if isinstance(descriptor, descriptor_module.DiftFpn):
        key_to_volume = extract_fpn_volumes(om.obs, descriptor, args.device)  # {key: (C_k, H, W)}
        strides, channels = dict(descriptor.strides), dict(descriptor.channels)
        del descriptor   # free SD-1.5 VRAM before clustering/matplotlib work
        figures = {
            tag: fpn_cluster_figure(om, {k: cluster_viz(v, num_clusters, seeds)
                                         for k, v in key_to_volume.items()},
                                    seeds, strides, channels,
                                    alpha=args.alpha, cmap=args.cmap, cluster_labels=legend,
                                    title=f"{title_base}  DiftFpn{tag.replace('_', '  ')}")
            for tag, seeds, legend in runs
        }
        base_name = f"clusters_fpn_{args.idx:04d}"
    else:
        volume = extract_descriptor_volume(om.obs, descriptor, args.device)  # (C, H, W)
        del descriptor   # free SD-1.5 VRAM before clustering/matplotlib work
        figures = {
            tag: cluster_figure(om, cluster_viz(volume, num_clusters, seeds), seeds,
                                alpha=args.alpha, cmap=args.cmap, cluster_labels=legend,
                                title=f"{title_base}{tag.replace('_', '  ')}")
            for tag, seeds, legend in runs
        }
        base_name = f"clusters_{args.idx:04d}"

    out_dir = args.render_dir.parent / f"{args.render_dir.name}_viz_clusters"
    for tag, fig in figures.items():
        out_path = (args.out.with_stem(args.out.stem + tag) if args.out
                    else out_dir / f"{base_name}{tag}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path, args.dpi)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
