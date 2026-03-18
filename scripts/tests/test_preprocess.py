"""
Unit tests for Node 2 — preprocess.py

"""

import json
import io
import pytest
from unittest.mock import patch, mock_open, MagicMock

import preprocess as pp


pp.ftfy = MagicMock()
pp.ftfy.fix_text = lambda text: text




def make_raw_line(en, es):
    return json.dumps({"translation": {"en": en, "es": es}}) + "\n"

VALID_EN = "The patient has a high fever and needs immediate treatment."
VALID_ES = "El paciente tiene fiebre alta y necesita tratamiento inmediato."
SHORT_EN = "Hi"
SHORT_ES = "Hola"
LONG_EN = " ".join(["word"] * 250)
LONG_ES = " ".join(["palabra"] * 250)


class TestCleanText:

    def test_strips_whitespace(self):
        assert pp.clean_text("  hello   world  ") == "hello world"

    def test_handles_empty_string(self):
        assert pp.clean_text("") == ""

    def test_collapses_tabs_and_newlines(self):
        assert pp.clean_text("hello\t\nworld") == "hello world"


"""
This is to check if it's a valid pair.
"""
class TestIsValidPair:

    def test_accepts_normal_pair(self):
        assert pp.is_valid_pair(VALID_EN, VALID_ES) is True

    def test_rejects_too_short_english(self):
        assert pp.is_valid_pair(SHORT_EN, VALID_ES) is False

    def test_rejects_too_short_spanish(self):
        assert pp.is_valid_pair(VALID_EN, SHORT_ES) is False

    def test_rejects_too_long_english(self):
        assert pp.is_valid_pair(LONG_EN, VALID_ES) is False

    def test_rejects_too_long_spanish(self):
        assert pp.is_valid_pair(VALID_EN, LONG_ES) is False

    def test_rejects_extreme_length_ratio(self):
        en = " ".join(["word"] * 100)
        es = " ".join(["palabra"] * 10)
        assert pp.is_valid_pair(en, es) is False

    def test_accepts_borderline_valid_ratio(self):
        en = " ".join(["word"] * 20)
        es = " ".join(["palabra"] * 10)
        assert pp.is_valid_pair(en, es) is True


class TestToInstruction:

    def test_en_to_es_format(self):
        result = pp.to_instruction("Hello world test sentence", "Hola mundo oración de prueba", "medical", "en_to_es")
        assert result["direction"] == "en_to_es"
        assert result["input"] == "Hello world test sentence"
        assert result["output"] == "Hola mundo oración de prueba"
        assert result["domain"] == "medical"
        assert "English to Spanish" in result["instruction"]

    def test_es_to_en_format(self):
        result = pp.to_instruction("Hello world test sentence", "Hola mundo oración de prueba", "legal", "es_to_en")
        assert result["direction"] == "es_to_en"
        assert result["input"] == "Hola mundo oración de prueba"
        assert result["output"] == "Hello world test sentence"
        assert "Spanish to English" in result["instruction"]


