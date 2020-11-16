
import json
from typing import Tuple, List



 
class Config:
    def __init__(self, config_json=None):
        self._config = {
            "target_size": (150, 150, 1),
            "batch_size": 64,
            "white_list_formats": ['png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'],
            "nb_epochs": 10,
            "seed": 22,
            "log_dir": "./logs",
            "background_samples": 1000, 
            "evaluation_samples": 500
        }
        if config_json is not None:
            self.load_json(config_json)


    def get_property(self, property_name):
        if property_name not in self._config.keys():
            return None

        return self._config[property_name]

    def set_property(self, property_name, value):
        self._config[property_name] = value

    
    def load_json(self, json_file):
        with open(json_file) as fp:
            config_dict = json.load(fp)
            self._config.update(config_dict)

    @property
    def target_size(self) -> Tuple[int, int, int]:
        return tuple(self.get_property('target_size'))[:3]

    @property
    def batch_size(self) -> int:
        return int(self.get_property('batch_size'))

    @property
    def white_list_formats(self) -> List[str]:
        return self.get_property('white_list_formats')

    @property
    def nb_epochs(self) -> int:
        return int(self.get_property('nb_epochs'))

    @property
    def seed(self) -> int:
        return int(self.get_property('seed'))

    @property
    def log_dir(self) -> str:
        return str(self.get_property('log_dir'))

    @property
    def background_samples(self) -> int:
        return int(self.get_property('background_samples'))

    @property
    def evaluation_samples(self) -> int:
        return int(self.get_property('evaluation_samples'))

config = Config()






