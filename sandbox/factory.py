"""
SandboxFactory — returns the correct sandbox backend based on config.
Agents call this; they never import Docker or subprocess directly.
"""

from __future__ import annotations

from typing import Literal

from sandbox.subprocess_sandbox import SubprocessSandbox


def create_sandbox(
    sandbox_type: Literal["docker", "subprocess", "e2b"] = "subprocess",
) -> SubprocessSandbox:
    """
    Return an uninitialised sandbox backend.
    Call .setup(workspace_files) before using it.
    """
    if sandbox_type == "docker":
        try:
            from sandbox.docker_sandbox import DockerSandbox

            return DockerSandbox()  # type: ignore[return-value]
        except RuntimeError as exc:
            print(f"[sandbox] Docker unavailable ({exc}), falling back to subprocess")
            return SubprocessSandbox()

    if sandbox_type == "e2b":
        # E2B integration placeholder — falls back to subprocess if not configured
        try:
            from sandbox.e2b_sandbox import E2BSandbox

            return E2BSandbox()  # type: ignore[return-value]
        except ImportError:
            print("[sandbox] E2B SDK not installed, falling back to subprocess")
            return SubprocessSandbox()

    return SubprocessSandbox()
