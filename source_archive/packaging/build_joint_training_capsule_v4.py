"""Project local metadata/SHA bindings, preserving algorithms and all numbers."""
from __future__ import annotations
import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "github_ready_v4/training_capsule"
SHA = re.compile(r"\b[0-9a-f]{64}\b")
PRIVATE = re.compile(rb"[A-Za-z]:[\\/]Users[\\/]|/mnt/[a-z]/Users/|/home/(?:misha|misha3)/")
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def project(value):
    if isinstance(value, dict): return {project(k): project(v) for k,v in value.items()}
    if isinstance(value, list): return [project(v) for v in value]
    if not isinstance(value, str): return value
    value = value.replace("\\", "/").replace(ROOT.as_posix(), "public_source_root").replace(ROOT.parent.as_posix(), "external_source_root")
    value = re.sub(r"/home/[^/:;\s'\"]+", "[external-home]", value)
    value = re.sub(r"(?i)[a-z]:/+Users/+[^/\s]+", "[local-user]", value)
    if value.startswith("/mnt/"): value = "[external-local-path]/" + value.rsplit("/",1)[-1]
    return value
def norm_ast(text):
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str): node.value = SHA.sub("<sha256>", project(node.value))
            return node
    return ast.dump(Normalize().visit(ast.parse(text)), include_attributes=False)
def project_source(text):
    # Source backslashes are code, not filesystem separators.
    text=text.replace(ROOT.as_posix(),"public_source_root").replace(ROOT.parent.as_posix(),"external_source_root")
    return re.sub(r"/home/[^/:;\s'\"]+", "[external-home]", text)
def tree_rebind(value,mapping):
    if isinstance(value, dict): return {tree_rebind(k,mapping):tree_rebind(v,mapping) for k,v in value.items()}
    if isinstance(value,list): return [tree_rebind(v,mapping) for v in value]
    if isinstance(value,str): return SHA.sub(lambda m:mapping.get(m.group(),m.group()),project(value))
    return value

if __name__ == "__main__":
    assert not OUT.exists(), "Preserve any prior capsule attempt"
    freeze = read(ROOT / "docs/joint_corner_training_v8/freeze.json")
    source_names = set(freeze["common_runtime_source_sha256"])
    template = read(ROOT / "docs/joint_corner_plans_v8/w8_control.bound.json")
    source_names.update(binding["path"] for binding in template["sources"].values())
    source_names.add(template["evidence"]["diagnosis"]["path"])
    source_names.add("artifacts/refinement/joint_v2/path_pilot111/path/final.npz")
    raw = {name:(ROOT / name).read_bytes() for name in sorted(source_names)}
    original_hashes = {name:sha_bytes(data) for name,data in raw.items()}
    for name,digest in freeze["common_runtime_source_sha256"].items(): assert original_hashes[name] == digest,name
    hash_to_name = {digest:name for name,digest in original_hashes.items()}
    representations = {}
    for name,data in raw.items():
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive: representations[name] = archive.read("data").decode()
        elif name.endswith(".npz"): representations[name] = ""
        else: representations[name] = data.decode("utf-8-sig")
    graph = {name:{hash_to_name[h] for h in SHA.findall(text) if h in hash_to_name} for name,text in representations.items()}
    # No expected SHA is made to lie about a file. Resolve actual projected bytes
    # in dependency order and bind their actual hashes. Cycles abort the export.
    pending, order = set(raw), []
    while pending:
        ready = sorted(n for n in pending if not graph[n].intersection(pending))
        if not ready: raise ValueError("Guard SHA dependency cycle: " + repr({n:sorted(graph[n]&pending) for n in pending}))
        order.extend(ready); pending.difference_update(ready)
    output_data, mapping, records = {}, {}, []
    for name in order:
        data = raw[name]
        changes = "none"
        if name.endswith(".zip"):
            buffer=io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as target:
                original_model=read_data=json.loads(source.read("data"))
                model=tree_rebind(original_model,mapping)
                model["bc_kl_debug_log_path"]="on_policy_prior_kl.jsonl"
                assert {k:v for k,v in model.items() if k!="bc_kl_debug_log_path"} == tree_rebind({k:v for k,v in original_model.items() if k!="bc_kl_debug_log_path"},mapping)
                for member in source.infolist():
                    payload=json.dumps(model,indent=2,ensure_ascii=False).encode() if member.filename=="data" else source.read(member.filename)
                    assert not PRIVATE.search(payload),(name,member.filename)
                    target.writestr(copy.copy(member),payload)
                    if member.filename!="data": assert payload == source.read(member.filename)
                for value in model.values():
                    if isinstance(value,dict) and ":serialized:" in value: assert not PRIVATE.search(base64.b64decode(value[":serialized:"]))
            result=buffer.getvalue(); changes="Only JSON path/SHA metadata; all other ZIP member bytes exact"
        elif name.endswith(".npz"):
            result=data
        elif name.endswith(".json"):
            original=json.loads(representations[name]); projected=tree_rebind(original,mapping)
            result=(json.dumps(projected,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
            changes="Only local path metadata and actual guard SHA rebindings; all scalar numbers/booleans unchanged"
        else:
            text=representations[name]
            projected=SHA.sub(lambda m:mapping.get(m.group(),m.group()),project_source(text))
            if name.endswith(".py"): assert norm_ast(text)==norm_ast(projected),(name,"algorithm AST changed")
            result=projected.encode(); changes="Only local path strings and actual guard SHA literal rebindings; normalized AST exact"
        assert not PRIVATE.search(result),name
        digest=sha_bytes(result); mapping[original_hashes[name]]=digest; output_data[name]=result
        records.append(dict(path=name, original_sha256=original_hashes[name], export_sha256=digest,
                            byte_identical=(data==result), transformation=changes, bytes=len(result)))
    OUT.mkdir(parents=True)
    for name,data in output_data.items():
        path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    # Original validated mappings form a template, not a copied approval or READY.
    projected_template=tree_rebind(template,mapping)
    public_template={k:projected_template[k] for k in ("schema","stage","contract","hypothesis","changed_factors","expected_effect","reconsider_if","sources","metric_alignment","development_environment_seeds","evidence","budget","argv")}
    public_template["status"]="DRAFT_FRESH_BASELINE_AND_SEMANTICS_REQUIRED"
    public_template["changed_factors"] += " Public portability metadata/SHA projection; numerical algorithm unchanged. Fresh baseline and semantic proof are mandatory in this workspace."
    path=OUT/"public_plan.template.json";path.write_text(json.dumps(public_template,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest=dict(schema="portable-v8-training-capsule-v1", status="EXPERIMENTAL_AWAIT_BASELINE_QA_NO_PPO_RUN",
        source_projection="Dependency-ordered actual SHA rebinding, never spoofed sha(); normalized Python AST and numeric JSON semantics preserved",
        current_measured_model_unchanged=True, initial_original_sha256=template["sources"]["initial_policy"]["sha256"],
        initial_export_sha256=mapping[template["sources"]["initial_policy"]["sha256"]],
        source_file_count=len(records), files=records, template_sha256=sha_bytes(path.read_bytes()),
        original_plan_sha256=sha_bytes((ROOT/"docs/joint_corner_plans_v8/w8_control.bound.json").read_bytes()),
        fresh_user_semantic_probe_required=True, actual_PPO_training_executed=False, actual_Gazebo_available=False)
    (OUT/"CAPSULE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    receipt=ROOT/"audit/github_packaging_v4_capsule_build_01.json"
    with receipt.open("x",encoding="utf-8") as handle:json.dump(manifest,handle,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=manifest["status"],files=len(records),bytes=sum(len(x) for x in output_data.values()),initial_export_sha256=manifest["initial_export_sha256"])))
