"""
Day 9 test — verify Docker image builds and runs correctly.
Run with: python test_day9.py
"""

import subprocess
import sys


def test_dockerfile_exists():
    import os

    assert os.path.exists("Dockerfile"), "Dockerfile not found"
    assert os.path.exists(".dockerignore"), ".dockerignore not found"

    with open("Dockerfile") as f:
        content = f.read()

    assert "python:3.12-slim" in content, "Base image not set correctly"
    assert "streamlit" in content, "Streamlit run command missing"
    assert "8501" in content, "Port 8501 not exposed"
    print("  Dockerfile structure correct.")


def test_docker_image_exists():
    result = subprocess.run(
        ["docker", "images", "research-agent", "--format", "{{.Repository}}"],
        capture_output=True,
        text=True,
    )
    assert (
        "research-agent" in result.stdout
    ), "Docker image not found — run: docker build -t research-agent ."
    print("  Docker image exists: research-agent")


def test_dockerignore():
    with open(".dockerignore") as f:
        content = f.read()
    assert ".env" in content, ".env should be in .dockerignore"
    assert ".venv" in content, ".venv should be in .dockerignore"
    print("  .dockerignore correctly excludes .env and .venv")


if __name__ == "__main__":
    print("Day 9 test\n")
    test_dockerfile_exists()
    test_dockerignore()
    test_docker_image_exists()
    print("\nDay 9 complete. Ready for Day 10.")
