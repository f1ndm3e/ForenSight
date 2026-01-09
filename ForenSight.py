import os, glob, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

Image.MAX_IMAGE_PIXELS = None
IMG_EXTS = ("*.png")

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def pick_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def list_images(folder):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(folder, ext))
    return sorted(paths)

def list_dump_dirs(root):
    return [p for p in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(p)]

def build_backbone(device):
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet

@torch.no_grad()
def embed_tiles(backbone, img_paths, device, batch_size=64):
    feats = []
    kept = []

    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        imgs = []
        ok = []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(TF(img))
                ok.append(p)
            except:
                pass

        if not imgs:
            continue

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
        h = backbone(x)
        h = h.view(h.size(0), 512)

        h = F.normalize(h, dim=1)

        feats.append(h.cpu())
        kept += ok

    if not feats:
        raise RuntimeError("No readable tiles.")
    return torch.cat(feats, dim=0), kept

def topk_dump_embedding(tile_embs, proto, topk=64):
    sims = (tile_embs @ proto.unsqueeze(1)).squeeze(1)
    k = min(topk, sims.shape[0])
    idx = torch.topk(sims, k=k).indices
    z = tile_embs[idx].mean(dim=0)
    z = F.normalize(z, dim=0)
    return z, sims

def save_npz(path, prototype, threshold, topk, train_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        prototype=prototype.cpu().numpy().astype(np.float32),
        threshold=np.float32(threshold),
        topk=np.int32(topk),
        train_names=np.array(train_names),
    )

def load_npz(path):
    art = np.load(path)
    proto = torch.from_numpy(art["prototype"]).float()
    thr = float(art["threshold"])
    topk = int(art["topk"]) if "topk" in art.files else 64
    train_names = art["train_names"].tolist() if "train_names" in art.files else []
    return proto, thr, topk, train_names

def cmd_train(train_root, out_npz, topk=64, batch_size=64, margin=1.05, force_cpu=False):
    device = pick_device(force_cpu)
    print(f"[+] Device: {device} (cuda_available={torch.cuda.is_available()})")

    backbone = build_backbone(device)
    dump_dirs = list_dump_dirs(train_root)
    if len(dump_dirs) < 2:
        raise SystemExit("[!] Need at least 2 dump folders inside train_root")

    dump_embs = []
    names = []

    # 1) compute mean of all tiles to seed prototype
    seed_embs = []
    for d in dump_dirs:
        imgs = list_images(d)
        if not imgs:
            continue
        tile_embs, _ = embed_tiles(backbone, imgs, device, batch_size=batch_size)
        z0 = F.normalize(tile_embs.mean(dim=0), dim=0)
        seed_embs.append(z0)
        names.append(os.path.basename(d))
        print(f"[+] Seed embed: {names[-1]} tiles={len(imgs)}")

    seed_embs = torch.stack(seed_embs, dim=0)
    proto0 = F.normalize(seed_embs.mean(dim=0), dim=0)

    # 2) recompute top-K pooling
    for d in dump_dirs:
        bn = os.path.basename(d)
        if bn not in names:
            continue
        imgs = list_images(d)
        tile_embs, _ = embed_tiles(backbone, imgs, device, batch_size=batch_size)
        z, _ = topk_dump_embedding(tile_embs, proto0, topk=topk)
        dump_embs.append(z)

    dump_embs = torch.stack(dump_embs, dim=0)
    proto = F.normalize(dump_embs.mean(dim=0), dim=0)

    # 3) set threshold from *actual* train distances
    dists = torch.norm(dump_embs - proto.unsqueeze(0), p=2, dim=1).cpu().numpy().tolist()
    thr = max(dists) * margin

    print("[+] Train distances:", [round(x, 6) for x in sorted(dists)])
    print(f"[+] Threshold (max*{margin}) = {thr:.6f}")

    save_npz(out_npz, proto.cpu(), thr, topk, names)
    print(f"[+] Saved -> {out_npz}")

def cmd_infer(tiles_dir, art_path, batch_size=64, show_top=64, force_cpu=False):
    #cuda for nvidia gpu
    device = pick_device(force_cpu)
    print(f"[+] Device: {device} (cuda_available={torch.cuda.is_available()})")

    proto, thr, topk, train_names = load_npz(art_path)
    proto = F.normalize(proto, dim=0)

    backbone = build_backbone(device)

    imgs = list_images(tiles_dir)
    tile_embs, kept = embed_tiles(backbone, imgs, device, batch_size=batch_size)

    dump_emb, sims = topk_dump_embedding(tile_embs, proto, topk=topk)
    dist = float(torch.norm(dump_emb - proto, p=2).item())
    verdict = "AGENTTESLA_MATCH" if dist <= thr else "NOT_MATCHING_OR_UNKNOWN"

    print(f"[+] Tiles read: {len(kept)}")
    print(f"[+] Distance to prototype: {dist:.6f}")
    print(f"[+] Threshold:            {thr:.6f}")
    print(f"[+] Verdict:              {verdict}")
    print(f"[+] Top {min(show_top, len(kept))} tiles (highest similarity):")

    kshow = min(show_top, sims.shape[0])
    top_idx = torch.topk(sims, k=kshow).indices.tolist()
    for rank, j in enumerate(top_idx, 1):
        print(f"    #{rank:02d}  sim={float(sims[j]):.6f}  file={os.path.basename(kept[j])}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--train_root", required=True, help="Folder containing subfolders (each subfolder = one dump's 625 tiles)")
    tr.add_argument("--out", required=True, help="Output .npz model path")
    tr.add_argument("--topk", type=int, default=64)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--margin", type=float, default=1.05)
    tr.add_argument("--cpu", action="store_true")

    inf = sub.add_parser("infer")
    inf.add_argument("--tiles_dir", required=True)
    inf.add_argument("--art", required=True)
    inf.add_argument("--batch", type=int, default=64)
    inf.add_argument("--show_top", type=int, default=64)
    inf.add_argument("--cpu", action="store_true")

    args = ap.parse_args()

    if args.cmd == "train":
        cmd_train(args.train_root, args.out, topk=args.topk, batch_size=args.batch, margin=args.margin, force_cpu=args.cpu)
    else:
        cmd_infer(args.tiles_dir, args.art, batch_size=args.batch, show_top=args.show_top, force_cpu=args.cpu)

if __name__ == "__main__":
    main()
