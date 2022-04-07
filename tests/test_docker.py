import base64
import json

import numpy as np
import pytest
import requests
import tensorflow as tf

SERVER_URL = (
    "https://doaa-ca2-emotive-model.herokuapp.com/v1/models/img_classifier:predict"
)

data = tf.keras.utils.image_dataset_from_directory(
    "tests/data/dataset",
    color_mode="rgb",
    batch_size=128,
    image_size=(48, 48),
    class_names=(
        "Anger",
        "Fear",
        "Surprise",
        "Happiness",
        "Neutral",
        "Sadness",
        "Disgust",
    ),
)

def make_prediction(instances):
    data = json.dumps(
        {"signature_name": "serving_default", "instances": instances.tolist()}
    )

    headers = {"content-type": "application/json"}

    json_response = requests.post(SERVER_URL, data=data, headers=headers)
    predictions = json.loads(json_response.text)["predictions"]
    return predictions


@pytest.mark.parametrize("data", [next(iter(data))])
def test_prediction(data, capsys):
    """Check that data performs as expected on images in the dataset."""
    target_acc = 60
    with capsys.disabled():
        if type(data) is tuple:
            data, expected = data
            expected = np.array(expected)
        else:
            expected = None
        data = np.array(data)
        predictions = make_prediction(data)
        assert len(predictions) == len(data)
        correct = 0
        for i, pred in enumerate(predictions):
            pred = np.array(pred)
            assert np.isclose(np.sum(pred), 1), "Softmax prediction should sum to 1"
            assert np.all(
                (pred >= 0) & (pred <= 1)
            ), "Softmax outputs should have a range of 0 to 1"
            if expected is not None:
                if expected[i] == np.argmax(pred):
                    correct += 1
        if expected is not None:
            acc = correct / len(predictions) * 100
            assert acc > target_acc
    return predictions


@pytest.mark.parametrize("data", [np.random.randn(64, 48, 48, 3)])
def test_random_noise(data, capsys):
    test_prediction(data, capsys)
    # TODO: Validity tests




@pytest.mark.xfail(reason="Invalid Data")
@pytest.mark.parametrize("data", [[True, "Hello World"]])
def test_invalid_data(data, capsys):
    """Check for valid response on invalid data"""
    test_prediction(data, capsys)


@pytest.mark.parametrize("data", [next(iter(data))])
def test_consistency(data, capsys):
    """Check if model is consistent when the same data is fed into it

    :param data: [description]
    :type data: [type]
    :param capsys: [description]
    :type capsys: [type]
    """
    with capsys.disabled():
        data, _ = data
        data = np.array(data)
        predictions1 = make_prediction(data)
        predictions2 = make_prediction(data)
        assert len(predictions1) == len(data)
        assert len(predictions2) == len(data)

        for pred1, pred2 in zip(predictions1, predictions2):
            assert np.allclose(pred1, pred2)
