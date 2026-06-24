import os

import torch


def get_cann_python_site_packages() -> str | None:
    """Return CANN Python site-packages if ``acl`` is importable from there."""
    candidates: list[str] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        base = os.environ.get(env_key)
        if not base:
            continue
        candidates.extend(
            [
                os.path.join(base, "python", "site-packages"),
                os.path.normpath(os.path.join(base, "..", "python", "site-packages")),
            ]
        )
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")

    for path in candidates:
        if os.path.isdir(os.path.join(path, "acl")):
            return path
    return None


def prepend_pythonpath(env: dict[str, str], *paths: str) -> None:
    existing = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    existing_parts = {part for part in existing.split(os.pathsep) if part}
    prefix_parts = [path for path in paths if path and path not in existing_parts]
    if prefix_parts:
        env["PYTHONPATH"] = os.pathsep.join([*prefix_parts, existing] if existing else prefix_parts)


def is_npu() -> bool:
    if not hasattr(torch, "npu"):
        return False

    if not torch.npu.is_available():
        raise RuntimeError("torch_npu detected, but NPU device is not available or visible.")

    return True
