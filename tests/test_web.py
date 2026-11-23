import json
import re

import pytest

from imu.api.main import WEB_DIR
from tests.conftest import CLASSES

EXAMPLES = WEB_DIR / "examples"


def test_root_redirects_to_the_web_app(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/app/"


def test_web_app_page_is_served(client):
    r = client.get("/app/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<title>Classification de trajets IMU</title>" in r.text


def test_web_app_does_not_expose_python_sources(client):
    assert client.get("/app/make_examples.py").status_code == 404
    assert client.get("/app/../make_examples.py").status_code == 404


def test_every_element_used_by_the_script_exists_in_the_page():
    html = (WEB_DIR / "index.html").read_text()
    used = set(re.findall(r'\$\("([\w-]+)"\)', html)) | set(re.findall(r'getElementById\("([\w-]+)"\)', html))
    defined = set(re.findall(r'\bid="([\w-]+)"', html))
    assert used and used <= defined, f"identifiants absents du HTML : {used - defined}"


def test_examples_cover_the_five_classes(client):
    manifest = client.get("/app/examples/index.json").json()
    assert [m["true_class"] for m in manifest] == CLASSES


@pytest.mark.parametrize(
    "entry", json.loads((EXAMPLES / "index.json").read_text()), ids=lambda e: e["true_class"]
)
def test_each_example_is_a_valid_api_request(client, entry):
    ex = client.get(f"/app/examples/{entry['file']}").json()
    assert ex["true_class"] == entry["true_class"]
    assert len(ex["rows"]) >= 256
    # même conversion que la page : colonnes + lignes -> mesures
    samples = [dict(zip(ex["columns"], row, strict=True)) for row in ex["rows"]]
    r = client.post("/v1/predict", json={"ride_id": ex["ride_id"], "samples": samples})
    assert r.status_code == 200, r.text
