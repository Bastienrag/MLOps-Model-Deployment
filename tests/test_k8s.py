"""Cohérence des manifestes Kubernetes entre eux et avec l'image."""

import re
from pathlib import Path

import pytest
import yaml

K8S = Path("k8s")


def load(name):
    return yaml.safe_load((K8S / name).read_text())


@pytest.fixture(scope="module")
def container():
    return load("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]


def test_three_replicas_with_zero_unavailable_during_updates():
    spec = load("deployment.yaml")["spec"]
    assert spec["replicas"] == 3
    assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_probes_use_health_on_the_api_port(container):
    for probe in ("readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"] == {"path": "/health", "port": 8000}


def test_liveness_waits_longer_than_readiness(container):
    """Un redémarrage est coûteux : la liveness attend plus que la readiness."""
    assert (
        container["livenessProbe"]["initialDelaySeconds"] > container["readinessProbe"]["initialDelaySeconds"]
    )


def test_requests_and_limits_are_set_and_consistent(container):
    res = container["resources"]
    assert set(res["requests"]) == set(res["limits"]) == {"cpu", "memory"}
    assert int(res["requests"]["memory"].removesuffix("Mi")) <= int(
        res["limits"]["memory"].removesuffix("Mi")
    )
    assert int(res["requests"]["cpu"].removesuffix("m")) <= int(res["limits"]["cpu"].removesuffix("m"))


def test_container_runs_as_the_non_root_user_of_the_image(container):
    uid = int(re.search(r"--uid (\d+)", Path("Dockerfile").read_text()).group(1))
    ctx = container["securityContext"]
    assert ctx["runAsNonRoot"] is True and ctx["runAsUser"] == uid
    assert ctx["allowPrivilegeEscalation"] is False and ctx["capabilities"] == {"drop": ["ALL"]}


def test_container_port_matches_the_image(container):
    assert container["ports"][0]["containerPort"] == 8000
    assert "EXPOSE 8000" in Path("Dockerfile").read_text()


def test_service_selects_the_pods_and_nodeport_is_published_by_kind():
    service = load("service.yaml")["spec"]
    labels = load("deployment.yaml")["spec"]["template"]["metadata"]["labels"]
    assert service["selector"].items() <= labels.items()
    assert service["type"] == "NodePort" and service["ports"][0]["targetPort"] == 8000
    mapping = load("kind-config.yaml")["nodes"][0]["extraPortMappings"][0]
    assert mapping["containerPort"] == service["ports"][0]["nodePort"]


def test_configuration_comes_from_the_configmap_not_the_image(container):
    config = load("configmap.yaml")
    assert container["envFrom"] == [{"configMapRef": {"name": config["metadata"]["name"]}}]
    # l'API lit bien ces variables
    api = Path("src/imu/api/main.py").read_text()
    assert "MODEL_URI" in api
    assert (
        config["data"]["MODEL_URI"] == "/app/models"
        and "ENV MODEL_URI=/app/models" in Path("Dockerfile").read_text()
    )


def test_no_credentials_in_the_manifests():
    text = "".join(p.read_text() for p in K8S.glob("*.yaml")).lower()
    assert not re.search(r"password|secret_key|access_key|token", text)
