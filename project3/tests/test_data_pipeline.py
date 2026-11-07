import unittest
import numpy as np
from train import split_indices, augment_batch, tensor_images


class PipelineTest(unittest.TestCase):
    def test_split_is_stratified_disjoint_and_reproducible(self):
        y = np.tile(np.arange(10),5000)
        train, val = split_indices(y)
        again,_ = split_indices(y)
        np.testing.assert_array_equal(again,train)
        self.assertEqual(len(np.intersect1d(train,val)),0)
        self.assertEqual(len(np.union1d(train,val)),50000)
        np.testing.assert_array_equal(np.bincount(y[val]),np.full(10,500))

    def test_crop_flip_matches_reference_loop(self):
        images = np.random.default_rng(123).integers(0,256,(5,32,32,3),dtype=np.uint8)
        pad = np.pad(images,((0,0),(4,4),(4,4),(0,0)),mode='reflect')
        ids = np.array([4,1,3])
        actual = augment_batch(pad,ids,np.random.default_rng(55))
        rng = np.random.default_rng(55)
        top = rng.integers(0,9,3); flip = rng.random(3)<.5; left = rng.integers(0,9,3)
        for k,idx in enumerate(ids):
            crop = pad[idx,top[k]:top[k]+32,left[k]:left[k]+32]
            np.testing.assert_array_equal(actual[k],crop[:,::-1] if flip[k] else crop)
        converted = tensor_images(actual,'cpu')
        self.assertEqual(tuple(converted.shape),(3,3,32,32))
        self.assertTrue(converted.is_contiguous())
