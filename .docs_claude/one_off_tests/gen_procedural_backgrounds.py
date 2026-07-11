"""Generates procedural background images (solid/pattern/noise) into random_backgrounds/procedural/.
Deterministic per-index seed, idempotent (skip existing unless --overwrite). See plan
`read-how-our-current-whimsical-kurzweil.md` S3."""

import argparse
from pathlib import Path

import torch
from torchvision.io import write_png

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "assets" / "random_backgrounds" / "procedural"


def solid_color(h: int, w: int) -> torch.Tensor:
    shade = torch.randint(0, 3, (1,)).item()   # occasionally force pure black/white/grey
    if shade == 0:
        color = torch.rand(3)
    else:
        color = torch.full((3,), float(torch.randint(0, 2, (1,)).item()))
    return color.view(3, 1, 1).expand(3, h, w).clone()


def checkerboard(h: int, w: int) -> torch.Tensor:
    tile = torch.randint(8, 96, (1,)).item()
    c1, c2 = torch.rand(3), torch.rand(3)
    row = torch.arange(h).view(h, 1) // tile
    col = torch.arange(w).view(1, w) // tile
    mask = ((row + col) % 2 == 0)
    out = torch.where(mask.unsqueeze(0), c1.view(3, 1, 1), c2.view(3, 1, 1))
    return out


def stripes(h: int, w: int) -> torch.Tensor:
    period = torch.randint(4, 128, (1,)).item()
    orientation = torch.randint(0, 3, (1,)).item()   # 0=horizontal, 1=vertical, 2=diagonal
    c1, c2 = torch.rand(3), torch.rand(3)
    row = torch.arange(h).view(h, 1)
    col = torch.arange(w).view(1, w)
    if orientation == 0:
        band = row // period
    elif orientation == 1:
        band = col // period
    else:
        band = (row + col) // period
    mask = (band % 2 == 0).expand(h, w)
    out = torch.where(mask.unsqueeze(0), c1.view(3, 1, 1), c2.view(3, 1, 1))
    return out


def gradient(h: int, w: int) -> torch.Tensor:
    c1, c2 = torch.rand(3), torch.rand(3)
    if torch.rand(1).item() < 0.5:
        axis = torch.randint(0, 2, (1,)).item()   # 0=vertical, 1=horizontal linear blend
        t = torch.linspace(0, 1, h if axis == 0 else w)
        t = t.view(-1, 1) if axis == 0 else t.view(1, -1)
        t = t.expand(h, w)
    else:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        r = torch.sqrt(yy**2 + xx**2)
        t = (r / r.max()).clamp(0, 1)
    t = t.unsqueeze(0)
    return c1.view(3, 1, 1) * (1 - t) + c2.view(3, 1, 1) * t


def uniform_noise(h: int, w: int) -> torch.Tensor:
    return torch.rand(3, h, w)


def gaussian_noise(h: int, w: int) -> torch.Tensor:
    mu, sigma = torch.rand(1).item(), 0.05 + 0.3 * torch.rand(1).item()
    return (torch.randn(3, h, w) * sigma + mu).clamp(0, 1)


FAMILIES = {
    "solid": solid_color,
    "checkerboard": checkerboard,
    "stripes": stripes,
    "gradient": gradient,
    "uniform": uniform_noise,
    "gaussian": gaussian_noise,
}


def write_family(name: str, fn, count: int, size: int, out_dir: Path, overwrite: bool, seed_base: int) -> tuple[int, int]:
    written, skipped = 0, 0
    for i in range(count):
        dst = out_dir / f"{name}_{i:05d}.png"
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        torch.manual_seed(seed_base + i)
        img = fn(size, size).clamp(0, 1)
        write_png((img * 255).round().to(torch.uint8), str(dst))
        written += 1
    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--count-solid", type=int, default=200)
    parser.add_argument("--count-checkerboard", type=int, default=150)
    parser.add_argument("--count-stripes", type=int, default=150)
    parser.add_argument("--count-gradient", type=int, default=150)
    parser.add_argument("--count-uniform", type=int, default=175)
    parser.add_argument("--count-gaussian", type=int, default=175)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    counts = {
        "solid": args.count_solid,
        "checkerboard": args.count_checkerboard,
        "stripes": args.count_stripes,
        "gradient": args.count_gradient,
        "uniform": args.count_uniform,
        "gaussian": args.count_gaussian,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    seed_base = 0
    total_written, total_skipped = 0, 0
    for name, fn in FAMILIES.items():
        written, skipped = write_family(name, fn, counts[name], args.size, args.out, args.overwrite, seed_base)
        print(f"{name}: {written} written, {skipped} skipped")
        total_written += written
        total_skipped += skipped
        seed_base += counts[name]

    print(f"total: {total_written} written, {total_skipped} skipped -> {args.out}")


if __name__ == "__main__":
    main()
