"""Manifest-driven multi-source background image downloader + convert-on-ingest.
HF sources stream via `datasets`, Kaggle via `kagglehub`, the ABO archive via a streamed
S3 tar (no full-archive download). Per image: convert to RGB, downscale to max side 1024,
save as quality-90 JPEG, dedupe by content hash of the saved bytes. See plan
`read-how-our-current-whimsical-kurzweil.md` S1-2.

Cannot be fully exercised here (no Kaggle token, multi-GB downloads) -- correctness of the
HF sources was checked with a live streaming probe; `# VERIFY` marks what wasn't. `datasets`
and `kagglehub` are imported lazily per-fetcher so e.g. HF-only runs don't need kagglehub."""

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "random_backgrounds"
MAX_SIDE = 1024
JPEG_QUALITY = 90

# No NSFW classifier: sources below are chosen SFW-by-construction (landscapes, scenery art,
# cartoon avatars, movie stills, product/room photos) per plan S2 point 2.
MANIFEST = [
    {
        "category": "landscapes",
        "subfolder": "landscapes",
        "access": "hf",
        "id": "ljnlonoljpiljm/places365-256px",
        "split": "train",
        "cap": 5000,
        "license_note": "Places365 research terms",
    },
    {
        "category": "anime",
        "subfolder": "anime",
        "access": "kaggle",
        # Swapped off HF `animelover/scenery-images` (dead under datasets>=3: legacy loading script over
        # multi-GB zip shards, raises "Dataset scripts are no longer supported"). This Kaggle set is
        # ~3980 768x768 anime-art images, fetched via kagglehub file rglob (no loading-script problem).
        "id": "imsahibnanda/anime-images-dataset-with-their-description",
        "cap": 4000,
        "license_note": "anime art, user-uploaded -> internal training only (SFW: verify at ingest)",
    },
    {
        "category": "cartoon",
        "subfolder": "cartoon",
        "access": "kaggle",
        # VERIFY: several Kaggle mirrors of Google's Cartoon Set exist
        # (imreallyjohn/cartoonset10k, brendanartley/cartoon-faces-googles-cartoon-set, ...);
        # this is the most-referenced slug as of writing -- confirm it resolves and carries the
        # CC BY 4.0 terms of the original Google release at first real run.
        "id": "imreallyjohn/cartoonset10k",
        "cap": 4000,
        "license_note": "CC BY 4.0 (Google Cartoon Set)",
    },
    {
        "category": "movies",
        "subfolder": "movies",
        "access": "kaggle",
        "id": "arka47/movie-frames-24k",
        "cap": 4000,
        "license_note": "copyrighted stills -> internal only",
    },
    {
        "category": "ikea",
        "subfolder": "ikea",
        "access": "hf",
        "id": "crawlfeeds/IKEA-Home-Decor-Furniture-Dataset",
        "split": "train",
        "cap": 3000,
        # Confirmed live: this repo is a product-listing CSV, not a blob Image feature.
        # `primary_image` holds an https:// URL, fetched over HTTP per row.
        "column_hint": "primary_image",
        "license_note": "IKEA.com property -> non-commercial only",
    },
    {
        "category": "amazon",
        "subfolder": "amazon",
        "access": "s3_tar",
        "id": "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar",
        "cap": 4000,
        # VERIFY: internal layout is images/small/<prefix>/<id>.jpg per the ABO docs; extraction
        # below filters tar members by extension so the exact prefix depth doesn't matter.
        "license_note": "CC BY-NC 4.0",
    },
]


def pick_image_column(features) -> str | None:
    import datasets as hf_datasets
    for name, feat in features.items():
        if isinstance(feat, hf_datasets.Image):
            return name
    return None


def download_url_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data))


def fetch_hf(entry: dict) -> Iterator[Image.Image]:
    from datasets import load_dataset
    ds = load_dataset(entry["id"], split=entry.get("split", "train"), streaming=True)
    image_col = pick_image_column(ds.features) or entry.get("column_hint")
    if image_col is None:
        raise ValueError(f"{entry['id']}: no Image feature and no column_hint in manifest")

    for example in ds.take(entry["cap"]):
        value = example[image_col]
        if isinstance(value, Image.Image):
            yield value
        elif isinstance(value, str) and value.startswith("http"):
            try:
                yield download_url_image(value)
            except Exception as e:
                print(f"  skip (url fetch failed): {value}: {e}")
        else:
            print(f"  skip (unrecognized cell in '{image_col}'): {value!r:.80}")


def fetch_kaggle(entry: dict) -> Iterator[Image.Image]:
    # kagglehub can't read the OAuth ~/.kaggle/credentials.json that `kaggle auth login` writes; the
    # kaggle CLI >=2.2 can. Run it project-independently via `uvx kaggle@latest` (`uv run --with kaggle`
    # breaks on the worktree's un-inited submodule deps). Kaggle has no partial download, so this pulls
    # the whole dataset into a staging dir on the assets' (big) disk, takes `cap`, then removes it.
    staging_base = ASSETS_ROOT.resolve().parent
    staging_base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_dl_", dir=str(staging_base)))
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    try:
        result = subprocess.run(
            ["uvx", "kaggle@latest", "datasets", "download", "-d", entry["id"], "-p", str(tmp), "--unzip"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,   # swallow the tqdm progress bar
        )
        if result.returncode != 0:
            tail = " | ".join((result.stderr or "").strip().splitlines()[-3:])
            print(f"  kaggle download failed ({entry['id']}): {tail}")
            return
        n = 0
        for p in sorted(tmp.rglob("*")):
            if n >= entry["cap"]:
                break
            if not p.is_file() or p.suffix.lower() not in valid_extensions:
                continue
            try:
                with Image.open(p) as im:
                    im.load()
                    img = im.copy()
                yield img
                n += 1
            except Exception as e:
                print(f"  skip (unreadable {p.name}): {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch_s3_tar(entry: dict) -> Iterator[Image.Image]:
    """Streams the tar sequentially (mode 'r|*'), never storing the full archive on disk."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    n = 0
    with urllib.request.urlopen(entry["id"]) as resp:
        with tarfile.open(fileobj=resp, mode="r|*") as tar:
            for member in tar:
                if n >= entry["cap"]:
                    break
                if not member.isfile() or Path(member.name).suffix.lower() not in valid_extensions:
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                try:
                    yield Image.open(io.BytesIO(fileobj.read()))
                    n += 1
                except Exception as e:
                    print(f"  skip (unreadable {member.name}): {e}")


FETCHERS = {"hf": fetch_hf, "kaggle": fetch_kaggle, "s3_tar": fetch_s3_tar}


def ingest(img: Image.Image, dst_dir: Path, stem: str, seen_hashes: set[str]) -> bool:
    """Convert-on-ingest one image: RGB, downscale to max side MAX_SIDE, save as jpg,
    dedupe by content hash of the saved bytes. Returns True if a new file was written."""
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = MAX_SIDE / max(w, h)
        if scale < 1:
            rgb = rgb.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
    except Exception as e:
        print(f"  skip (convert failed): {e}")
        return False

    digest = hashlib.sha1(data).hexdigest()
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)
    (dst_dir / f"{stem}_{digest[:12]}.jpg").write_bytes(data)
    return True


def existing_hashes(dst_dir: Path) -> set[str]:
    return {hashlib.sha1(p.read_bytes()).hexdigest() for p in dst_dir.glob("*.jpg")}


def run_source(entry: dict, out_root: Path, limit: int | None) -> tuple[int, int]:
    cap = limit if limit is not None else entry["cap"]
    dst_dir = out_root / entry["subfolder"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen = existing_hashes(dst_dir)

    written = skipped = 0
    try:
        for img in FETCHERS[entry["access"]]({**entry, "cap": cap}):
            if ingest(img, dst_dir, entry["category"], seen):
                written += 1
            else:
                skipped += 1
    except Exception as e:
        print(f"[{entry['category']}] FETCH FAILED: {type(e).__name__}: {e}")
    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=[e["category"] for e in MANIFEST])
    parser.add_argument("--limit", type=int, default=None, help="override every source's cap (smoke run)")
    parser.add_argument("--out", type=Path, default=ASSETS_ROOT)
    args = parser.parse_args()

    entries = [e for e in MANIFEST if args.only is None or e["category"] in args.only]
    for entry in entries:
        print(f"[{entry['category']}] fetching from {entry['id']} ({entry['access']}) ...")
        written, skipped = run_source(entry, args.out, args.limit)
        print(f"[{entry['category']}] {written} written, {skipped} skipped -> {args.out / entry['subfolder']}")

    # HF streaming leaves aiohttp/pyarrow worker threads that crash at interpreter finalization
    # (PyGILState_Release). All work is done + printed above, so skip the broken teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
