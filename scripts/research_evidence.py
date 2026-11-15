"""Stdlib-only immutable research-data check/restore; never runs training."""
from pathlib import Path,PurePosixPath
import argparse
import hashlib
import json
import os
import re
import stat
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST=ROOT/'evidence/research_evidence_dependency_manifest_v24_candidate.json'
EMBEDDED='_DEPENDENCY_MANIFEST.json'

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def safe_name(value):
    if not isinstance(value,str) or not value or chr(92) in value or ':' in value:raise ValueError('Canonical relative POSIX path required')
    path=PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(p in ('','..','.') for p in path.parts) or str(path)!=value:raise ValueError('Unsafe archive path')
    return path

def load_manifest(path):
    data=Path(path).read_bytes();manifest=json.loads(data)
    if manifest['schema']!='research-evidence-dependencies-v1':raise ValueError('Unsupported manifest schema')
    rows={}
    for row in manifest['files']:
        name=str(safe_name(row['path']))
        if name in rows or name==EMBEDDED:raise ValueError('Duplicate/reserved manifest member')
        if not isinstance(row['bytes'],int) or isinstance(row['bytes'],bool) or row['bytes']<0:raise ValueError('Invalid byte size')
        if not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('Invalid SHA256')
        rows[name]=row
    prefix=str(safe_name(manifest['archive_prefix']))+'/'
    if len(rows)!=manifest['file_count'] or sum(r['bytes'] for r in rows.values())!=manifest['total_file_bytes']:raise ValueError('Manifest total mismatch')
    return manifest,data,rows,prefix

def check_root(root,rows):
    root=Path(root).resolve();checked=0
    for name,row in rows.items():
        target=root.joinpath(*safe_name(name).parts)
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):raise ValueError('Missing/unsafe required input: '+name)
        if target.stat().st_size!=row['bytes'] or sha_file(target)!=row['sha256']:raise ValueError('Input mismatch: '+name)
        checked+=1
    return checked

def inspect_archive(path,manifest_bytes,rows,prefix,destination=None):
    staging=None
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist();names=[i.filename for i in infos]
        expected={prefix+n for n in rows}|{prefix+EMBEDDED}
        if len(names)!=len(set(names)) or set(names)!=expected:raise ValueError('Archive does not exactly match manifest members')
        for info in infos:
            safe_name(info.filename)
            mode=info.external_attr>>16
            if info.is_dir() or stat.S_IFMT(mode) not in (0,stat.S_IFREG) or info.flag_bits&1:raise ValueError('Only regular unencrypted files accepted')
            expected_size=len(manifest_bytes) if info.filename==prefix+EMBEDDED else rows[info.filename[len(prefix):]]['bytes']
            if info.file_size!=expected_size:raise ValueError('Unexpected expanded member size')
        if archive.read(prefix+EMBEDDED)!=manifest_bytes:raise ValueError('Embedded manifest differs from trusted local manifest')
        if destination is not None:
            destination=Path(destination).resolve()
            if destination.exists():raise FileExistsError('Restore requires a fresh destination')
            destination.parent.mkdir(parents=True,exist_ok=True)
            staging=destination.parent/(destination.name+'.partial_'+uuid.uuid4().hex)
            staging.mkdir()
        for name,row in rows.items():
            digest=hashlib.sha256();written=0;output=None
            try:
                if staging is not None:
                    target=staging.joinpath(*safe_name(name).parts);target.parent.mkdir(parents=True,exist_ok=True);output=target.open('xb')
                with archive.open(prefix+name) as source:
                    for chunk in iter(lambda:source.read(1024*1024),b''):
                        written+=len(chunk)
                        if written>row['bytes']:raise ValueError('Member exceeds manifest bound')
                        digest.update(chunk)
                        if output is not None:output.write(chunk)
                if written!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('Archive payload mismatch: '+name)
            finally:
                if output is not None:output.close()
        if staging is not None:
            if destination.exists():raise FileExistsError('Destination appeared during restore; partial output preserved')
            os.rename(staging,destination)
    return len(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--check-root',type=Path)
    mode.add_argument('--check-archive',type=Path)
    mode.add_argument('--restore',type=Path,metavar='ARCHIVE')
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args();manifest,data,rows,prefix=load_manifest(args.manifest)
    result=dict(status='PLAN_ONLY',snapshot=manifest['snapshot'],manifest_status=manifest['status'],files=len(rows),bytes=sum(r['bytes'] for r in rows.values()),manifest_sha256=hashlib.sha256(data).hexdigest(),training_or_simulation_started=False)
    if args.check_root is not None:
        result.update(status='PASS_REQUIRED_INPUT_BYTES',checked_files=check_root(args.check_root,rows))
    elif args.check_archive is not None:
        result.update(status='PASS_ARCHIVE_MEMBERS_CRC_SHA256',checked_files=inspect_archive(args.check_archive,data,rows,prefix))
    elif args.restore is not None:
        if args.destination is None:parser.error('--restore requires --destination pointing to a fresh directory')
        result.update(status='PASS_RESTORED_EXACT_INPUT_BYTES',checked_files=inspect_archive(args.restore,data,rows,prefix,args.destination))
    elif args.destination is not None:parser.error('--destination applies only to --restore')
    result['scope']='Byte integrity only; restored import/preflight/runtime completeness and scientific replication are separate checks'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
