"""Small original-pixel/profile fixtures; weight loading is mocked, no real models."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from utils import original299,models
from utils.io import read_json,sha256,write_json


def fixture(root,mode='RGB',size=(299,299)):
    original=root/'raw.png';Image.new(mode,size).save(original)
    canonical=root/'canonical.png';Image.new('RGB',(224,224)).save(canonical)
    record={'id':'a','label':3,'original_label':3,'split':'pilot','target_label':7,
        'original_path':str(original),'original_sha256':sha256(original),'original_size':list(size),
        'canonical_path':str(canonical),'canonical_sha256':sha256(canonical)}
    manifest=root/'canonical.json'
    write_json(manifest,{'schema_version':1,'counts':{'pilot':1,'dev':0,'holdout':0},
        'label_offset':0,'records':[record],'preprocessing':{'resize':[224,224]}})
    return manifest,record


def args(manifest,**changes):
    return argparse.Namespace(manifest=str(manifest),source=changes.get('source','rn50'),
        ftm_input_profile=changes.get('ftm_input_profile','canonical224-v2'),
        ftm_layer_profile=changes.get('ftm_layer_profile','mapped299'))


class Original299Tests(unittest.TestCase):
    def test_prepare_references_exact_original_no_resize_or_new_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest,record=fixture(root);output=root/'data/original299'
            before=sha256(record['original_path'])
            with patch.object(original299,'ROOT',root),patch('PIL.Image.Image.resize',side_effect=AssertionError('resize forbidden')):
                result=original299.prepare(manifest,output,expected_count=1)
            saved=read_json(result['manifest']);row=saved['records'][0]
            self.assertEqual(row['canonical_path'],record['original_path'])
            self.assertEqual(row['canonical_sha256'],before);self.assertEqual(sha256(record['original_path']),before)
            self.assertEqual(saved['source_manifest_sha256'],sha256(manifest))
            self.assertEqual(saved['counts'],{'pilot':1,'dev':0,'holdout':0})
            self.assertEqual((row['label'],row['split'],row['target_label']),(3,'pilot',7))
            self.assertEqual(list(output.iterdir()),[output/'manifest.json'])
            self.assertEqual(result['model_calls'],0);self.assertEqual(result['pixel_files_written'],0)
            with patch.object(original299,'ROOT',root):
                with self.assertRaises(FileExistsError):original299.prepare(manifest,output,expected_count=1)
                with self.assertRaises(ValueError):original299.prepare(manifest,root/'results/wrong',expected_count=1)

    def test_original_size_rgb_and_sha_rejected(self):
        for mode,size in [('RGBA',(299,299)),('RGB',(224,224))]:
            with self.subTest(mode=mode,size=size),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);manifest,_=fixture(root,mode,size)
                with patch.object(original299,'ROOT',root):
                    with self.assertRaisesRegex(ValueError,'RGB299'):original299.prepare(manifest,root/'data/new',1)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest,record=fixture(root)
            Image.new('RGB',(299,299),(1,2,3)).save(record['original_path'])
            with patch.object(original299,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'SHA'):original299.prepare(manifest,root/'data/new',1)

    def test_count_unique_and_source_identity_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest,record=fixture(root)
            with patch.object(original299,'ROOT',root):
                with self.assertRaises(ValueError):original299.prepare(manifest,root/'data/new')
                original299.prepare(manifest,root/'data/new',1)
            source=root/'data/new/manifest.json';saved=read_json(source)
            settings=args(source,source='rn50_v1',ftm_input_profile='original299-v1')
            spec=original299.input_spec(settings,saved['records'])
            self.assertEqual((spec['input_size'],spec['qualification_size']),(299,299))
            changed=copy.deepcopy(saved);changed['records'][0]['label']=4;write_json(source,changed)
            with self.assertRaisesRegex(ValueError,'identity'):original299.input_spec(settings,changed['records'])
            write_json(source,saved);source_old=read_json(manifest);source_old['new_field']='drift';write_json(manifest,source_old)
            with self.assertRaisesRegex(ValueError,'source manifest changed'):original299.input_spec(settings,saved['records'])
            duplicate=copy.deepcopy(source_old);duplicate['records']=[record,record];write_json(manifest,duplicate)
            with patch.object(original299,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'duplicate'):original299.prepare(manifest,root/'data/duplicate',2)

    def test_explicit_profiles_mutually_exclusive_and_old_identity_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest,record=fixture(root)
            old=original299.input_spec(args(manifest),[record]);identity={'name':'rn50','input_size':[224,224],'preprocessing':'old unchanged'}
            self.assertIs(original299.model_identity(identity,old),identity)
            self.assertEqual(old['profile_component'],'local224');self.assertEqual(old['input_size'],224)
            self.assertEqual(original299.input_spec(args(manifest,ftm_layer_profile='native224'),[record])['qualification_size'],224)
            with self.assertRaises(ValueError):original299.input_spec(args(manifest,source='rn50_v1'),[record])
            with patch.object(original299,'ROOT',root):original299.prepare(manifest,root/'data/new',1)
            manifest=root/'data/new/manifest.json';records=read_json(manifest)['records']
            for source,layer in [('rn50','mapped299'),('rn50_v1','native224')]:
                with self.subTest(source=source,layer=layer),self.assertRaises(ValueError):
                    original299.input_spec(args(manifest,source=source,ftm_input_profile='original299-v1',ftm_layer_profile=layer),records)
            with self.assertRaises(ValueError):original299.input_spec(args(manifest),records)
            spec=original299.input_spec(args(manifest,source='rn50_v1',ftm_input_profile='original299-v1'),records)
            new=original299.model_identity({'name':'rn50_v1','weights':'IMAGENET1K_V1'},spec)
            self.assertEqual(new['input_size'],[299,299]);self.assertIn('no resize',new['preprocessing'])
            Image.new('RGB',(224,224)).save(records[0]['original_path'])
            with self.assertRaises(ValueError):original299.input_spec(args(manifest,source='rn50_v1',ftm_input_profile='original299-v1'),records)

    def test_model_registry_uses_explicit_local_v1_and_retains_v2(self):
        from torchvision import models as vision
        with tempfile.TemporaryDirectory() as temporary:
            cache=Path(temporary);folder=cache/'hub/checkpoints';folder.mkdir(parents=True)
            for name,weights in [('rn50_v1',vision.ResNet50_Weights.IMAGENET1K_V1),('rn50',vision.ResNet50_Weights.IMAGENET1K_V2)]:
                with self.subTest(name=name):
                    checkpoint=folder/weights.url.rsplit('/',1)[-1];checkpoint.write_bytes(b'mock local official checkpoint')
                    network=nn.Linear(1,1)
                    with patch.object(models,'TORCH_HOME',cache),patch.object(vision,'resnet50',return_value=network) as constructor,patch('torch.load',return_value=network.state_dict()) as load:
                        wrapped,identity=models.load_model(name,'cpu')
                    constructor.assert_called_once_with(weights=None)
                    load.assert_called_once_with(checkpoint,map_location='cpu',weights_only=True)
                    self.assertEqual(identity['name'],name);self.assertEqual(identity['weights'],str(weights))
                    self.assertEqual(identity['input_size'],[224,224]);self.assertFalse(wrapped.training)
                    self.assertTrue(all(not p.requires_grad for p in wrapped.parameters()))
                    if name=='rn50_v1':self.assertTrue(identity['url'].endswith('resnet50-0676ba61.pth'))
            with patch.object(models,'TORCH_HOME',cache/'missing'):
                with self.assertRaises(FileNotFoundError):models.load_model('rn50_v1','cpu')
