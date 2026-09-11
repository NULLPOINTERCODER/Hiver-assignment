"""Unit tests for Intent Classifiers and Predictor."""
import pytest
from src.intent.baseline import MajorityBaselineClassifier, TfidfLogisticClassifier
from src.intent.classifier import SemanticIntentClassifier
from src.intent.predictor import IntentPredictor

@pytest.fixture
def dummy_data():
    X = [
        "My battery is dying fast and draining",
        "Battery percentage drops to zero",
        "How do I update to iOS 11?",
        "Software update is stuck on Apple logo",
        "My Apple ID password is locked",
        "Account security compromise"
    ]
    y = [
        "battery_power_issue",
        "battery_power_issue",
        "software_update_issue",
        "software_update_issue",
        "account_security_issue",
        "account_security_issue"
    ]
    return X, y

def test_majority_baseline(dummy_data):
    X, y = dummy_data
    clf = MajorityBaselineClassifier().fit(X, y)
    preds = clf.predict(["Random query", "Another query"])
    assert len(preds) == 2
    assert preds[0] in clf.classes_

def test_tfidf_baseline(dummy_data):
    X, y = dummy_data
    clf = TfidfLogisticClassifier().fit(X, y)
    pred = clf.predict(["My battery is draining quickly"])
    assert pred[0] == "battery_power_issue"

def test_semantic_classifier_and_predictor(dummy_data, tmp_path):
    X, y = dummy_data
    clf = SemanticIntentClassifier().fit(X, y)
    
    # Test prediction
    pred = clf.predict(["Need help with my Apple ID locked"])
    assert pred[0] == "account_security_issue"
    
    # Test serialization
    saved_path = clf.save(tmp_path)
    loaded_clf = SemanticIntentClassifier.load(saved_path)
    loaded_pred = loaded_clf.predict(["Need help with my Apple ID locked"])
    assert loaded_pred[0] == "account_security_issue"

    # Test predictor interface
    predictor = IntentPredictor(loaded_clf, confidence_threshold=0.50)
    res = predictor.predict_one("Battery drained quickly")
    assert "intent" in res
    assert "confidence" in res
    assert "top_intents" in res
