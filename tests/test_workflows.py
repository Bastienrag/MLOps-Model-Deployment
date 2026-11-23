"""Règles de sécurité des workflows. Un runner auto-hébergé sur un dépôt public ne doit exécuter que du
code de confiance : ces tests échouent si une modification ouvre une brèche."""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = {p.name: yaml.safe_load(p.read_text()) for p in Path(".github/workflows").glob("*.yml")}


def triggers(workflow):
    return workflow.get("on", workflow.get(True))  # PyYAML lit la clé `on` comme le booléen True


def uses(workflow):
    return [s["uses"] for job in workflow["jobs"].values() for s in job["steps"] if "uses" in s]


@pytest.mark.parametrize("name", WORKFLOWS)
def test_actions_are_pinned_to_a_commit(name):
    for action in uses(WORKFLOWS[name]):
        assert re.fullmatch(r".+@[0-9a-f]{40}", action), (
            f"{action} n'est pas épinglée par empreinte de commit"
        )


@pytest.mark.parametrize("name", WORKFLOWS)
def test_permissions_are_restricted_and_jobs_have_a_timeout(name):
    workflow = WORKFLOWS[name]
    assert workflow["permissions"].get("contents") == "read"
    assert "write" not in str(workflow["permissions"]).replace("read", "")
    for job_name, job in workflow["jobs"].items():
        assert "timeout-minutes" in job, f"{name}/{job_name} sans timeout"


def test_only_the_deploy_job_runs_on_the_self_hosted_runner():
    for name, workflow in WORKFLOWS.items():
        for job_name, job in workflow["jobs"].items():
            self_hosted = "self-hosted" in str(job["runs-on"])
            assert self_hosted == (name == "deploy.yml" and job_name == "deploy"), f"{name}/{job_name}"


def test_ci_never_runs_on_pull_request_code_with_the_runner():
    assert "pull_request" in triggers(WORKFLOWS["ci.yml"])
    assert "pull_request" not in triggers(WORKFLOWS["deploy.yml"])


def test_deploy_only_follows_a_completed_ci_on_main():
    on = triggers(WORKFLOWS["deploy.yml"])["workflow_run"]
    assert on == {"workflows": ["CI"], "types": ["completed"], "branches": ["main"]}


def test_deploy_is_gated_by_a_guard_that_requires_a_green_push_on_main():
    jobs = WORKFLOWS["deploy.yml"]["jobs"]
    assert jobs["deploy"]["needs"] == "guard"
    assert "needs.guard.outputs.allowed == 'true'" in jobs["deploy"]["if"]
    script = jobs["guard"]["steps"][0]["run"]
    for condition in (
        '"$RUN_CONCLUSION" = "success"',
        '"$RUN_EVENT" = "push"',
        '"$RUN_BRANCH" = "main"',
        '"$RUN_REPO" = "$REPO"',
    ):
        assert condition in script
    assert "self-hosted" not in str(jobs["guard"]["runs-on"])


def test_deploy_rolls_back_on_failure():
    steps = WORKFLOWS["deploy.yml"]["jobs"]["deploy"]["steps"]
    script = next(s["run"] for s in steps if s.get("name", "").startswith("Déploiement sur kind"))
    assert "kubectl rollout status" in script and "kubectl rollout undo" in script and "exit 1" in script


def test_docker_job_waits_for_lint_tests_and_gate_and_publishes_only_from_main():
    docker = WORKFLOWS["ci.yml"]["jobs"]["docker"]
    assert set(docker["needs"]) == {"lint-test", "model-gate"}
    assert "refs/heads/main" in docker["env"]["PUBLISH"] and "'push'" in docker["env"]["PUBLISH"]


def test_secrets_are_never_exposed_to_a_whole_job():
    """Un secret passé tel quel dans `env` d'un job serait lisible par chacune de ses étapes."""
    for name, workflow in WORKFLOWS.items():
        for job_name, job in workflow["jobs"].items():
            for key, value in job.get("env", {}).items():
                assert not re.fullmatch(r"\$\{\{\s*secrets\.\w+\s*\}\}", str(value)), (
                    f"{name}/{job_name} : {key}"
                )
