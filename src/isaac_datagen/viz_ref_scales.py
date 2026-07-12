import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F

from isaac_datagen.viz_clusters import alpha_crop
from vision_core.datastructs import ObsMaskDescriptorMetadata
from vision_core.viz import fit_pca_basis, panel_grid, pca_rgb, rgba_chw_to_rgb, save_figure


def scale_pca_rgb(feats, basis, min_hw):
    h, w = feats.shape[-2:]
    s = max(1.0, min_hw / min(h, w))
    up = F.interpolate(feats[None].float(), scale_factor=s,
                       mode="bilinear", align_corners=False)[0]
    return pca_rgb(up, basis)


def load_rows(render_dir, backbone):
    md = ObsMaskDescriptorMetadata
    c2r = md.deserialize_field(0, render_dir, "class_to_ref")
    c2d = md.deserialize_field(0, render_dir, "class_to_descriptors")[backbone]
    pcs = md.deserialize_field(0, render_dir, "principal_components")[backbone]
    assert isinstance(next(iter(c2d.values())), dict), \
        f"{backbone} is single-scale in {render_dir}; this tool needs a keyed multi-scale FPN backbone"
    ds = render_dir.parent.name
    return [(f"{ds}/{cls}", c2r[cls], c2d[cls], pcs) for cls in sorted(c2d)]


def joint_basis(rows):
    scales = list(next(iter(rows))[2])
    return {k: fit_pca_basis(
                torch.cat([leaf[k].flatten(1).T for _, _, leaf, _ in rows], dim=0), n=3)
            for k in scales}


def ref_scales_figure(rows, min_hw):
    scales = list(next(iter(rows))[2])
    fig, axes = panel_grid(len(rows) * (1 + len(scales)), cols=1 + len(scales))
    it = iter(axes)
    for label, ref, leaf, pcs in rows:
        a = next(it); a.imshow(rgba_chw_to_rgb(alpha_crop(ref)[0]))
        a.set_title(label, fontsize=8); a.axis("off")
        for k in scales:
            f = leaf[k]; a = next(it)
            a.imshow(scale_pca_rgb(f, pcs[k], min_hw))
            a.set_title(f"scale {k}  {f.shape[1]}×{f.shape[2]}  C={f.shape[0]}", fontsize=8)
            a.axis("off")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("render_dirs", type=Path, nargs="+")
    p.add_argument("--backbone", required=True)
    p.add_argument("--min-hw", type=int, default=96)
    p.add_argument("--joint-basis", action="store_true",
                   help="pool ALL rows' tokens per scale → columns comparable across datasets")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    rows = [row for rd in a.render_dirs for row in load_rows(rd, a.backbone)]
    if a.joint_basis:
        jb = joint_basis(rows)
        rows = [(label, ref, leaf, jb) for label, ref, leaf, _ in rows]
    ds_names = "+".join(dict.fromkeys(rd.parent.name for rd in a.render_dirs))
    suffix = "_joint" if a.joint_basis else ""
    out = a.out or (a.render_dirs[0].parent.parent / f"ref_scales_{ds_names}{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(ref_scales_figure(rows, a.min_hw), out, a.dpi)
    print(f"wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
