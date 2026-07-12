from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from isaac_datagen.runtime_config import load_config
from isaac_datagen.scene import boot_sim
from isaac_datagen.objects import GraspableObject, OptFlowObject, UsdPath
from isaac_datagen.graspableobj_to_optflow_obj import render_one


def main():
    cfg, in_dir, idx, out_png = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    runtime = load_config(cfg, [f"dataset_dir={out_png.parent}", "mode=optflow"])
    K = np.load(runtime.intrinsics_path).astype(np.float32)
    W, H = runtime.width, runtime.height

    app = boot_sim(runtime, out_png.parent)
    import omni.replicator.core as rep
    from PIL import Image as PILImage

    obj = GraspableObject.deserialize(idx, in_dir)
    rgb, depth, ref_pose_cv = render_one(app, rep, obj, K, W, H, runtime)
    o = OptFlowObject(
        usd_path=UsdPath(str(obj.usd_path)), meta=obj.meta,
        reference_image=PILImage.fromarray(rgb), reference_depth=depth,
        ref_intrinsics=K, ref_pose=ref_pose_cv.astype(np.float32),
        grasp_point=obj.grasp_point.astype(np.float32),
    )
    PILImage.fromarray(o.visualize(title=f"{obj.meta['name']} (centroid-aligned)")).save(out_png)
    print(f"wrote {out_png}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
