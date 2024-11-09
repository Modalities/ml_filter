import os
from pathlib import Path

import pytest

from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_dir = Path(os.path.dirname(__file__))


@pytest.fixture
def classifier_training_pipeline():
    return ClassifierTrainingPipeline(working_dir / "resources" / "configs" / "test_config.yaml")

@pytest.mark.skip(reason="Clasifier pipeline is not implemented yet")
def test_train_classifier(classifier_training_pipeline):
    try:
        # Act: Call the train method
        classifier_training_pipeline.train_classifier()
    except Exception as e:
        # Fail the test if any exception occurs
        pytest.fail(f"Training raised an unexpected exception: {e}")
