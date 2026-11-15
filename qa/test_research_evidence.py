"""Meaningful stdlib integrity/path/restore tests; tiny synthetic data only."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import stat
import tempfile
import unittest
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evidence_helper',ROOT/'scripts/research_evidence.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'runs').mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='synthetic_evidence_',dir=ROOT/'runs')
        self.root=Path(self.temporary.name).resolve();assert self.root.is_relative_to((ROOT/'runs').resolve())
        self.payloads={'src/example.py':b'# synthetic only\n','data/table.csv':b'x,y\n1,2\n'}
        self.manifest=dict(schema='research-evidence-dependencies-v1',status='SYNTHETIC_SOFTWARE_TEST',snapshot='fixture',archive_prefix='fixture',
            file_count=2,total_file_bytes=sum(map(len,self.payloads.values())),
            files=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())for n,b in self.payloads.items()])
        self.manifest_path=self.root/'manifest.json';self.manifest_path.write_text(json.dumps(self.manifest))
        self.m,self.data,self.rows,self.prefix=helper.load_manifest(self.manifest_path)
    def tearDown(self):self.temporary.cleanup()
    def archive(self,changes=None,extra=None,duplicate=False,symlink=False,embedded=None):
        path=self.root/'fixture.zip';members={self.prefix+n:b for n,b in self.payloads.items()}
        members[self.prefix+helper.EMBEDDED]=self.data if embedded is None else embedded
        members.update(changes or {});members.update(extra or {})
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name,b in members.items():
                info=zipfile.ZipInfo(name);info.external_attr=((stat.S_IFLNK if symlink and name.endswith('example.py') else stat.S_IFREG)|0o644)<<16
                z.writestr(info,b)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');z.writestr(self.prefix+'src/example.py',self.payloads['src/example.py'])
        return path
    def test_valid_archive_restore_and_tree(self):
        path=self.archive();self.assertEqual(helper.inspect_archive(path,self.data,self.rows,self.prefix),2)
        dest=self.root/'restored';self.assertEqual(helper.inspect_archive(path,self.data,self.rows,self.prefix,dest),2)
        self.assertEqual(helper.check_root(dest,self.rows),2)
        for name,b in self.payloads.items():self.assertEqual((dest/name).read_bytes(),b)
    def test_tampered_payload_is_rejected(self):
        path=self.archive(changes={self.prefix+'data/table.csv':b'x,y\n9,2\n'})
        with self.assertRaises(ValueError):helper.inspect_archive(path,self.data,self.rows,self.prefix)
    def test_traversal_member_is_rejected(self):
        path=self.archive(extra={'fixture/../../escape':b'bad'})
        with self.assertRaises(ValueError):helper.inspect_archive(path,self.data,self.rows,self.prefix,self.root/'restored')
        self.assertFalse((self.root/'restored').exists())
    def test_duplicate_member_is_rejected(self):
        with self.assertRaises(ValueError):helper.inspect_archive(self.archive(duplicate=True),self.data,self.rows,self.prefix)
    def test_symlink_member_is_rejected(self):
        with self.assertRaises(ValueError):helper.inspect_archive(self.archive(symlink=True),self.data,self.rows,self.prefix)
    def test_existing_destination_is_preserved(self):
        dest=self.root/'existing';dest.mkdir();(dest/'user.txt').write_text('preserve')
        with self.assertRaises(FileExistsError):helper.inspect_archive(self.archive(),self.data,self.rows,self.prefix,dest)
        self.assertEqual((dest/'user.txt').read_text(),'preserve')
    def test_changed_embedded_manifest_is_rejected(self):
        changed=self.data.replace(b'fixture',b'changed')
        with self.assertRaises(ValueError):helper.inspect_archive(self.archive(embedded=changed),self.data,self.rows,self.prefix)
    def test_changed_expanded_size_is_rejected(self):
        with self.assertRaises(ValueError):helper.inspect_archive(self.archive(changes={self.prefix+'data/table.csv':b'longer payload'}),self.data,self.rows,self.prefix)
    def test_manifest_unsafe_paths_are_rejected(self):
        for path in ('../a','/absolute','C:/drive','a'+chr(92)+'b','a//b','.'):
            with self.subTest(path=path),self.assertRaises(ValueError):helper.safe_name(path)
    def test_restored_tamper_and_escape_are_rejected(self):
        dest=self.root/'restored';helper.inspect_archive(self.archive(),self.data,self.rows,self.prefix,dest)
        (dest/'data/table.csv').write_bytes(b'x,y\n9,2\n')
        with self.assertRaises(ValueError):helper.check_root(dest,self.rows)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report=dict(status='PASS_STDLIB_SYNTHETIC_HELPER_QA' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        helper_sha256=helper.sha_file(ROOT/'scripts/research_evidence.py'),real_research_data_copied=0,
        real_research_archives_created=0,new_environment_steps=0,optimizer_steps=0)
    path=ROOT/'qa/research_evidence_helper_qa.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
