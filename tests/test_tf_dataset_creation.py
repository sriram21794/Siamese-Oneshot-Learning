import unittest
from dataset import get_tf_dataset

class TestTFDatasetCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = "./data/images_evaluation/"

    def test_tf_dataset_creation(self):
        self.tf_dataset = get_tf_dataset(self.directory)
        image_pairs, target = next(self.tf_dataset.as_numpy_iterator())
        
        self.assertIsInstance(image_pairs, dict)

        self.assertIn("input_1", image_pairs.keys())
        self.assertIn("input_2", image_pairs.keys())

        self.assertEqual(image_pairs['input_1'].shape[0], image_pairs['input_2'].shape[0])
        self.assertEqual(image_pairs['input_1'].shape[0], target.shape[0])
            

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestTFDatasetCreation('test_tf_dataset_creation'))
    runner = unittest.TextTestRunner()
    runner.run(suite)