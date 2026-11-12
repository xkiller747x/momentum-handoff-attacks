"""Reference original RGB PNG pixels without resize or model calls."""
import copy
from pathlib import Path

from PIL import Image

from .io import read_json,sha256,write_json

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_PREPROCESSING={'input_profile':'original299-v1','input_size':[299,299],
    'resize':None,'interpolation':None,'pixel_source':'original RGB299 PNG, referenced unchanged',
    'quantization':'original uint8 PNG pixels; no new image encoding',
    'protocol':'Original299 pixel/source audit; normalization only at model wrapper, not full author environment reproduction'}


def verify_original(record):
    path=Path(record['original_path']).resolve()
    if sha256(path)!=record['original_sha256']:raise ValueError(f'Original image SHA changed: {record["id"]}')
    with Image.open(path) as image:
        if image.format!='PNG' or image.mode!='RGB' or image.size!=(299,299):
            raise ValueError(f'Original must be RGB299x299 PNG: {record["id"]}; got {image.format}/{image.mode}/{image.size}')
        image.load()
    return path


def prepare(manifest_path,output,expected_count=1000):
    source=Path(manifest_path).resolve();output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'data').resolve()):raise ValueError('Original299 preparation must be inside project data/')
    destination=output/'manifest.json'
    if destination.exists():raise FileExistsError('Original299 manifest already exists; choose a new output')
    manifest=read_json(source);records=manifest['records'];ids=[r['id'] for r in records]
    if len(records)!=expected_count or len(ids)!=len(set(ids)):raise ValueError('Unexpected original299 record count or duplicate IDs')
    if any(not 0<=r['label']<1000 or r['split'] not in ('pilot','dev','holdout') for r in records):raise ValueError('Invalid labels or split')
    counts={split:sum(r['split']==split for r in records) for split in ('pilot','dev','holdout')}
    if manifest['counts']!=counts:raise ValueError('Source manifest split counts disagree')
    prepared=copy.deepcopy(manifest)
    for record in prepared['records']:
        path=verify_original(record)
        record.update(canonical_path=str(path),canonical_sha256=record['original_sha256'],original_size=[299,299])
    prepared.update(input_profile='original299-v1',source_manifest_path=str(source),source_manifest_sha256=sha256(source),
        preprocessing=copy.deepcopy(ORIGINAL_PREPROCESSING),counts=counts)
    write_json(destination,prepared)
    return {'manifest':str(destination),'count':len(records),'counts':counts,'model_calls':0,
        'pixel_files_written':0,'manifest_sha256':sha256(destination)}


def prepare_command(args):return prepare(args.manifest,args.output)


def input_spec(args,records):
    """Validate explicit FTM profile before loading a model; old defaults unchanged."""
    profile=getattr(args,'ftm_input_profile','canonical224-v2')
    if profile=='canonical224-v2':
        if args.source!='rn50':raise ValueError('canonical224-v2 requires source rn50 (V2)')
        manifest=read_json(args.manifest)
        if manifest.get('input_profile')=='original299-v1':raise ValueError('Original299 manifest requires original299-v1 input profile')
        for record in records:
            with Image.open(record['canonical_path']) as image:
                if image.mode!='RGB' or image.size!=(224,224):raise ValueError('canonical224-v2 requires RGB224 inputs')
        return {'input_size':224,'qualification_size':299 if args.ftm_layer_profile=='mapped299' else 224,
            'profile_component':'local224',
            'adaptation':'RN50V2/local224. mapped299 fixes exact module eligibility from a299 zero-input source forward; native224 uses current-output rule. Batch1 SelfShuffle self-mixes; batch2 fixed adjacent IDs, not author batch20.'}
    if profile!='original299-v1' or args.source!='rn50_v1' or args.ftm_layer_profile!='mapped299':
        raise ValueError('original299-v1 requires source rn50_v1 and layer profile mapped299')
    manifest=read_json(args.manifest)
    if manifest.get('input_profile')!='original299-v1' or manifest.get('preprocessing')!=ORIGINAL_PREPROCESSING:raise ValueError('Expected verified original299 manifest preprocessing')
    source=Path(manifest['source_manifest_path'])
    if sha256(source)!=manifest['source_manifest_sha256']:raise ValueError('Original299 source manifest changed')
    original=read_json(source);source_rows={r['id']:r for r in original['records']}
    if len(source_rows)!=len(original['records']) or len(manifest['records'])!=len(source_rows) or {r['id'] for r in manifest['records']}!=set(source_rows) or manifest['counts']!=original['counts']:raise ValueError('Original299 source cohort/counts differ')
    for record in manifest['records']:
        old=source_rows[record['id']]
        for key in ('original_path','original_sha256','id','label','split','original_label'):
            if record.get(key)!=old.get(key):raise ValueError('Original299 source record identity differs: '+record['id'])
        if Path(record['canonical_path']).resolve()!=Path(old['original_path']).resolve() or record['canonical_sha256']!=old['original_sha256']:raise ValueError('Original299 canonical reference differs from original')
    for record in records:verify_original(record)
    return {'input_size':299,'qualification_size':299,'profile_component':'original299-v1',
        'adaptation':'RN50 IMAGENET1K_V1/original299 RGB PNG, no resize. mapped299 eligibility and actual input both299. Strict opt-in exceeds author cudnn-only flag; fixed batch2 SelfShuffle differs from author batch20. No full author software/global-RNG reproduction.',
        'source_manifest_sha256':manifest['source_manifest_sha256'],'preprocessing':copy.deepcopy(ORIGINAL_PREPROCESSING)}


def model_identity(identity,spec):
    if spec['input_size']==224:return identity
    return {**identity,'input_size':[299,299],
        'preprocessing':'normalization only on verified original RGB299 uint8 PNG; no resize',
        'input_profile':'original299-v1'}
