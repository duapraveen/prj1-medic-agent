import medic_agent.rag.entity_extractor as extractor_module


def test_extract_entities_parses_valid_json(mocker):
    mocker.patch.object(
        extractor_module,
        "complete",
        return_value=(
            '{"entities": [{"text": "Type 2 diabetes", "entity_type": "Diagnosis"}]}',
            {},
        ),
    )
    result = extractor_module.extract_entities("Patient has type 2 diabetes.")
    assert result == [{"text": "Type 2 diabetes", "entity_type": "Diagnosis"}]


def test_extract_entities_strips_markdown_fences(mocker):
    mocker.patch.object(
        extractor_module,
        "complete",
        return_value=(
            '```json\n{"entities": [{"text": "Metformin", "entity_type": "Medication"}]}\n```',
            {},
        ),
    )
    result = extractor_module.extract_entities("Patient takes Metformin.")
    assert result == [{"text": "Metformin", "entity_type": "Medication"}]


def test_extract_entities_returns_empty_on_bad_json(mocker):
    mocker.patch.object(extractor_module, "complete", return_value=("not valid json", {}))
    result = extractor_module.extract_entities("some clinical text")
    assert result == []


def test_extract_entities_filters_items_with_non_string_fields(mocker):
    mocker.patch.object(
        extractor_module,
        "complete",
        return_value=(
            '{"entities": [{"text": 999, "entity_type": "Diagnosis"},'
            ' {"text": "Hypertension", "entity_type": "Diagnosis"}]}',
            {},
        ),
    )
    result = extractor_module.extract_entities("...")
    assert result == [{"text": "Hypertension", "entity_type": "Diagnosis"}]


def test_extract_entities_uses_entity_extractor_model(mocker):
    from medic_agent.config.settings import ENTITY_EXTRACTOR_MODEL_ID

    mock_complete = mocker.patch.object(
        extractor_module, "complete", return_value=('{"entities": []}', {})
    )
    extractor_module.extract_entities("some text")
    model_used = mock_complete.call_args[0][0]
    assert model_used == ENTITY_EXTRACTOR_MODEL_ID


def test_extract_entities_truncates_long_input(mocker):
    mock_complete = mocker.patch.object(
        extractor_module, "complete", return_value=('{"entities": []}', {})
    )
    long_text = "x" * 5000
    extractor_module.extract_entities(long_text)
    user_input_sent = mock_complete.call_args[0][2]
    assert len(user_input_sent) <= 2000
