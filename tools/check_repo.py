# Pre-publication check: no personal paths/names, no files > 50 MB, no model/audio/dll files. Exit code 1 on problems.
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BS = chr(92)
BAD_NAMES = [("Vol" + "pe"), "C:" + BS + "Users", "C:/" + "Users", "G:" + BS, "E:" + BS, "E:/" + "Prog", "Progetti" + " CLAUDE",
             "comfyui" + "-rocm", "god" + "nig"]
BAD_EXT = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".dll", ".npy", ".exe"}
problems = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    rel = path.relative_to(ROOT)
    if path.stat().st_size > 50 * 1024 * 1024:
        problems.append(f"{rel}: larger than 50 MB")
    if path.suffix.lower() in BAD_EXT or "__pycache__" in path.parts:
        problems.append(f"{rel}: forbidden file type or cache")
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for bad in BAD_NAMES:
        author_ok = bad == ("Vol" + "pe") and path.name in ("LICENSE", "README.md", "README.it.md")  # author credit allowed
        if bad in text and path.name != "check_repo.py" and not author_ok:
            problems.append(f"{rel}: contains '{bad}'")
    if re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text) and path.name not in ("check_repo.py",):
        for match in re.findall(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text):
            problems.append(f"{rel}: email-like text '{match}'")
print("\n".join(problems) if problems else "OK: no personal paths, no big files, no models/audio/dll")
sys.exit(1 if problems else 0)
