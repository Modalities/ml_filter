from pathlib import Path


class ClassifierClient:
    def __init__(self, config_file_path: Path):
        self.config_file_path = config_file_path

    def classify(self, text):
        pass
