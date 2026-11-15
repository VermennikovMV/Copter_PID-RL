"""Verify the publication allowlist, exact hashes and public text hygiene."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def generated_path(relative):
    return (any(part in (".git", ".venv", "__pycache__", ".pytest_cache", "runs") for part in relative.parts)
            or relative.parts[:3] == ("demo_three", "force_plugin", "build")
            or relative.as_posix() == "v2/artifacts/refinement/joint_v2/.gazebo_native_queue.lock")

def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in manifest["files"]}
    actual = {str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*") if p.is_file()
              and not generated_path(p.relative_to(ROOT))}
    assert actual == set(expected) | {"MANIFEST.json"}, (sorted(actual - set(expected)), sorted(set(expected) - actual))
    total = 0
    for name, entry in expected.items():
        path = ROOT / name
        data = path.read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], name
        assert len(data) < 50 * 1024 * 1024, name
        assert not re.search(r"(^|/)(\.env($|\.)|id_rsa|credentials|token)(/|$)", name, re.I), name
        assert not re.search(r"9300[123]", name), name
        if path.suffix in (".zip", ".npz"):
            with zipfile.ZipFile(path) as archive:
                for entry in archive.namelist():
                    assert not re.search(rb"[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/Users/|/home/(?:misha|misha3)/", archive.read(entry)), (name, entry)
        if path.suffix.lower() in (".md", ".json", ".py", ".yml", ".csv", ".txt"):
            content = data.decode("utf-8-sig")
            assert not re.search(r"[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/Users/|/home/(?:misha|misha3)/", content), name
            assert not re.search(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,})", content), name
            if path.suffix == ".md":
                for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                    if not re.match(r"(?:https?://|#|mailto:)", link):
                        assert (path.parent / link.split("#")[0]).exists(), (name, link)
        total += len(data)
    print(json.dumps({"status": "PASS", "files": len(expected), "bytes": total,
                      "manifest_self_hash_excluded": True}, indent=2))

if __name__ == "__main__":
    main()
