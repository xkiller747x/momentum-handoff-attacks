"""Reference-gallery preparation and independent CSV target-label joining."""
import csv
from pathlib import Path
from PIL import Image
from .io import read_json,sha256,write_json


def target_labels(path,records):
    result={}
    with open(path,newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not {'filename','label','targeted_label'}.issubset(reader.fieldnames or []):raise ValueError('Target CSV requires filename,label,targeted_label')
        for row in reader:
            identity=Path(row['filename']).stem;label=int(row['label']);target=int(row['targeted_label'])
            if not identity or identity in result:raise ValueError('Duplicate/empty target image ID')
            if not 0<=label<1000 or not 0<=target<1000 or label==target:raise ValueError('Invalid true/target labels')
            result[identity]={'label':label,'target':target}
    for record in records:
        if record['id'] not in result or result[record['id']]['label']!=record['label']:raise ValueError('Target CSV ID/true-label mismatch')
    return {record['id']:result[record['id']]['target'] for record in records}


def reference_files(images_dir,expected_per_class=10,classes=1000):
    root=Path(images_dir).resolve();result=[];seen=set();labels=set()
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():continue
        try:label=int(directory.name)
        except ValueError as error:raise ValueError('Reference class directory must be numeric') from error
        if not 0<=label<classes or label in labels:raise ValueError('Duplicate/out-of-range numeric reference class')
        labels.add(label)
        paths=sorted(p for p in directory.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        if len(paths)!=expected_per_class:raise ValueError(f'Reference class {label} expected {expected_per_class}, got {len(paths)}')
        for path in paths:
            identity=f'{label:04d}_{path.name}'
            if identity in seen:raise ValueError('Duplicate reference ID')
            seen.add(identity);result.append((label,identity,path))
    if labels!=set(range(classes)):raise ValueError('Incomplete reference class coverage')
    return sorted(result,key=lambda row:(row[0],row[1]))


def reference_prepare_command(args):
    output=Path(args.output).resolve()
    if (output/'manifest.json').exists():raise FileExistsError('Reference preparation already exists')
    rows=[]
    for label,identity,path in reference_files(args.images_dir,args.expected_per_class):
        destination=output/'images'/f'{identity}.png';destination.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(path) as image:
            size=list(image.size);image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR).save(destination)
        rows.append({'id':identity,'label':label,'original_path':str(path),'original_sha256':sha256(path),
            'original_size':size,'canonical_path':str(destination),'canonical_sha256':sha256(destination)})
    manifest={'records':rows,'count':len(rows),'class_count':1000,'expected_per_class':args.expected_per_class,
        'images_dir':str(Path(args.images_dir).resolve()),'label_mapping':'int(directory.name); never lexicographic ImageFolder classes',
        'preprocessing':'Pillow RGB bilinear full-image resize224x224, uint8 lossless PNG; ImageNetV2 MatchedFrequency local adaptation'}
    write_json(output/'manifest.json',manifest)
    return {'manifest':str(output/'manifest.json'),'count':len(rows),'model_forwards':0}


def reference_gallery(path):
    manifest=read_json(path);rows=manifest['records'];groups={label:[] for label in range(1000)};seen=set()
    for row in rows:
        if row['id'] in seen or row['label'] not in groups:raise ValueError('Invalid reference manifest ID/class')
        seen.add(row['id']);groups[row['label']].append(row)
    if any(len(values)!=10 for values in groups.values()):raise ValueError('Reference manifest requires exactly10 candidates for every class')
    return {label:sorted(values,key=lambda row:row['id']) for label,values in groups.items()}
