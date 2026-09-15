# -*- coding: utf-8 -*-
"""Human-readable dump of a .pt file.

Usage:
    python view_pt.py <file.pt>            # print to terminal
    python view_pt.py <file.pt> --save     # also write <file.pt>.txt next to it
"""
import sys
import torch


def describe(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}dict with {len(obj)} keys:")
        for k, v in obj.items():
            if hasattr(v, "shape"):  # a tensor
                print(f"{pad}  {str(k):45s} tensor{tuple(v.shape)} {v.dtype}")
            elif isinstance(obj, dict) and isinstance(v, (dict, list)):
                print(f"{pad}  {k}:")
                describe(v, indent + 2)
            else:
                print(f"{pad}  {str(k):45s} = {v!r}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__} of len {len(obj)}")
        for i, v in enumerate(obj[:20]):
            print(f"{pad}  [{i}] {type(v).__name__}"
                  + (f" shape{tuple(v.shape)}" if hasattr(v, "shape") else f" = {v!r}"))
    elif hasattr(obj, "shape"):
        print(f"{pad}tensor{tuple(obj.shape)} {obj.dtype}")
    else:
        print(f"{pad}{type(obj).__name__}: {obj!r}")


def main():
    if len(sys.argv) < 2:
        print("usage: python view_pt.py <file.pt> [--save]")
        return
    path = sys.argv[1]
    obj = torch.load(path, map_location="cpu", weights_only=False)

    import io
    buf = io.StringIO()
    _stdout = sys.stdout
    sys.stdout = buf
    print(f"=== {path} ===")
    print(f"top-level type: {type(obj).__name__}")
    describe(obj)
    sys.stdout = _stdout
    text = buf.getvalue()
    print(text)

    if "--save" in sys.argv:
        out = path + ".txt"
        with open(out, "w") as f:
            f.write(text)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
