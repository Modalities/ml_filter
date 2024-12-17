import unittest

import torch
from datasets import Dataset

from inference import collate_fn


class TestInferencePipeline(unittest.TestCase):
    def test_collate_fn(self):
        batch = [
            {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]},
            {'input_ids': [4, 5], 'attention_mask': [1, 1]}
        ]
        result = collate_fn(batch)

        expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        expected_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        self.assertTrue(torch.equal(result['input_ids'], expected_input_ids))
        self.assertTrue(torch.equal(result['attention_mask'], expected_attention_mask))

    def test_dataset_sharding(self):
        data = {'input_ids': [[1], [2], [3], [4]], 'attention_mask': [[1], [1], [1], [1]]}
        dataset = Dataset.from_dict(data)
        shard_0 = dataset.shard(num_shards=2, index=0)
        shard_1 = dataset.shard(num_shards=2, index=1)

        self.assertEqual(len(shard_0), 2)
        self.assertEqual(len(shard_1), 2)
        self.assertEqual(shard_0[0]['input_ids'], [1])
        self.assertEqual(shard_1[0]['input_ids'], [2])


if __name__ == '__main__':
    unittest.main()
