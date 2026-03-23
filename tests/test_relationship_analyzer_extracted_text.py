from unittest.mock import Mock

from sqlalchemy.orm import Session

from app.services.relationship_analyzer import RelationshipAnalyzer


def test_extract_stylistic_features_uses_extracted_text_field():
    mock_db_session = Mock(spec=Session)
    analyzer = RelationshipAnalyzer(mock_db_session)

    mock_document = Mock(id=1)
    mock_pages = [
        Mock(extracted_text="Alpha beta.", text=None),
        Mock(extracted_text="Gamma delta", text=None),
    ]

    mock_query = mock_db_session.query.return_value
    mock_query.get.return_value = mock_document
    mock_query.filter.return_value.all.return_value = mock_pages

    features = analyzer._extract_stylistic_features(1)

    assert features["word_count"] == 4
    assert features["sentence_count"] == 1
    assert features["text_length"] == len("Alpha beta. Gamma delta")
