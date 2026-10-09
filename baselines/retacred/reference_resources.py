"""One-command pinned source/parser/weight retrieval; no silent fallback."""
from __future__ import annotations
import json,os,ssl,tarfile,tempfile,shutil
from pathlib import Path,PurePosixPath
from urllib.parse import quote
from .common import ROOT,PACKAGE,atomic_json,digest
from .assets import stream_download,assemble,CA,reject_symlinks

LOCK=PACKAGE/'reference_resources.lock.json'

def fetch_file(row,root):
    reject_symlinks(root);root.mkdir(parents=True,exist_ok=True)
    name=row['name']
    if Path(name).name!=name:raise ValueError('unsafe resource name')
    target=root/name
    if target.exists():
        if target.is_symlink() or digest(target)!=row['sha256'] or target.stat().st_size!=row['size_bytes']:raise ValueError('corrupt resource cache preserved')
        from .reference_storage import clean_verified_parts
        clean_verified_parts(target,row,root/'parts')
        return target
    url=os.environ.get('PROJECT_EXCHANGE_URL','');token=os.environ.get('PROJECT_EXCHANGE_TOKEN','')
    if not url.startswith('https://') or not token:raise ValueError('configure HTTPS project exchange before missing asset fetch')
    context=ssl.create_default_context(cafile=CA)
    parts=[]
    for p in row['parts']:
        if Path(p['name']).name!=p['name']:raise ValueError('unsafe shard name')
        path=root/'parts'/p['name']
        stream_download(url.rstrip('/')+'/files/'+row['namespace']+'/'+quote(p['name']),path,expected=p['sha256'],size=p['size_bytes'],token=token,context=context)
        parts.append(path)
    assemble(parts,target,row['sha256'],row['size_bytes'])
    from .reference_storage import clean_verified_parts
    clean_verified_parts(target,row,root/'parts')
    return target

def prepare(models):
    lock=json.loads(LOCK.read_text());cache=ROOT/'.external/assets/reference_resources'/lock['revision']
    package=fetch_file(lock['source_bundle'],cache)
    sources=cache/'sources'
    if not sources.exists():
        stage=Path(tempfile.mkdtemp(prefix='sources.stage-',dir=cache))
        try:
            with tarfile.open(package) as archive:
                names=[]
                for member in archive:
                    p=PurePosixPath(member.name)
                    if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name not in lock['source_files']:
                        raise ValueError('undeclared/unsafe source archive member')
                    names.append(member.name);destination=stage/member.name;destination.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as source,destination.open('wb') as out:shutil.copyfileobj(source,out)
                if len(names)!=len(set(names)) or set(names)!=set(lock['source_files']):raise ValueError('incomplete/duplicate source bundle')
            verify_sources(stage,lock);stage.rename(sources)
        finally:
            if stage.exists():shutil.rmtree(stage)
    verify_sources(sources,lock)
    assets={name:str(sources/path) for name,path in lock['source_directories'].items()}
    if any(m in ('curriculum','luke_cfie') for m in models):
        wheel=fetch_file(lock['parser_asset'],cache)
        # Explicit source-independent parser data extraction; dependency package
        # installation is separately documented, never pip during formal runs.
        import zipfile
        parser=cache/'parser'
        if not parser.exists():
            stage=Path(tempfile.mkdtemp(prefix='parser.stage-',dir=cache))
            try:
                with zipfile.ZipFile(wheel) as z:
                    for n in z.namelist():
                        p=PurePosixPath(n)
                        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe parser wheel path')
                    z.extractall(stage)
                stage.rename(parser)
            finally:
                if stage.exists():shutil.rmtree(stage)
        assets['parser']=str(parser/'en_core_web_sm'/'en_core_web_sm-3.7.1')
    if any(not m.startswith(('luke_','ctl_drp_')) and m!='metaentail' for m in models):
        from .assets import fetch
        assets['roberta']=str(fetch())
    if any(m.startswith('luke_') for m in models):
        from .luke_assets import fetch
        assets['luke']=str(fetch())
    if 'metaentail' in models:
        from .mnli_assets import fetch
        assets['mnli']=str(fetch())
    for variant in ('sup','unsup'):
        if 'ctl_drp_'+('supervised' if variant=='sup' else 'unsupervised') in models:
            assets['ctl_'+variant]=str(fetch_file(lock['ctl_weights'][variant],cache/variant))
    assets['resource_lock_sha256']=digest(LOCK)
    assets['verified_assets']={name:{p.name:digest(p) for p in Path(path).iterdir() if p.is_file()}
        for name,path in assets.items() if name in ('roberta','luke','mnli')}
    assets['verified_assets'].update({name:digest(Path(path)) for name,path in assets.items() if name in ('ctl_sup','ctl_unsup')})
    return assets

def verify_sources(directory,lock):
    for name,sha in lock['source_files'].items():
        path=directory/name;reject_symlinks(path)
        if not path.is_file() or digest(path)!=sha:raise ValueError('pinned source bundle hash drift: '+name)
