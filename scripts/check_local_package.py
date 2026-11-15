"""Stdlib-only local docs/assets/provenance check; no NN, physics or training."""
from pathlib import Path
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    linked=0
    for p in [ROOT/'README.md',ROOT/'MODEL_CARD.md',*sorted((ROOT/'reference_manual').glob('*.md'))]:
        text=p.read_text(encoding='utf-8-sig')
        links=re.findall(r'\[[^\]]*\]\(([^)]+)\)',text)+re.findall(r'<img[^>]+src="([^"]+)"',text)
        for name in links:
            if re.match(r'(?:https?://|#|mailto:)',name):continue
            assert (p.parent/name.split('#')[0]).exists(),(p.relative_to(ROOT).as_posix(),name)
            linked+=1
    for p in (ROOT/'assets').rglob('*.svg'):ET.parse(p)
    pinned={
        'models/route_memory20_zero_integral.npz':'f0e87341177d9f2e3f2522973dd9887cf583736618aeb39ec850e98058ee8461',
        'v2/artifacts/refinement/joint_v2/metric_cost_pilot_v16_111/metric_cost/best_policy.npz':'68dafd9809c10c55dd82835a4464973017be2d3516aebae919f16f5a4676bcb5',
        'examples/memory20_encoded_obs64.npz':'feceb462ec6eef8efbb976700618694e88b94bf329f7f8c42169f612990cbc00',
        'assets/replays/legacy_hover_3way.gif':'22f37faacf14f2460aa3cd539c25d920adb473b93b26fd64b62825de4cf9623e',
        'assets/replays/f0_strong_route.gif':'88088a32bddb441963c051ed8a775ad9c0a236a6eb9f7f37e0716c3e072a5df0'}
    for name,digest in pinned.items():assert sha(ROOT/name)==digest,name
    index=read(ROOT/'evidence/presentation06_sources.json')
    for row in index['files']:assert sha(ROOT/row['path'])==row['export_sha256'],row['path']
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        frozen=read(manifest)
        for row in frozen['files']:
            p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    result=dict(status='PASS_LOCAL_METADATA_CHECKS',package_stage='FROZEN_MANIFEST_PRESENT' if manifest.exists() else 'PREP_ONLY_NO_FINAL_MANIFEST',
        local_links_checked=linked,pinned_policy_example_media_files=len(pinned),presentation_provenance_rows=len(index['files']),
        scientific_quality_or_reproduction_claim=False,new_NN_forwards=0,new_environment_steps=0,optimizer_steps=0)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
