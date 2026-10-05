import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def shapes() -> dict:
    return json.loads((FIXTURES / "kalshi" / "shapes.json").read_text())
