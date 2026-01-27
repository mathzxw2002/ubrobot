# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup


def get_version_from_toml() -> str:
    """Return the project's version string parsed from `pyproject.toml`.

    The function scans `pyproject.toml` line-by-line looking for a line
    that starts with ``version`` (for example: ``version = "1.2.3"``)
    and returns the value without surrounding quotes. If no such line is
    found a :class:`ValueError` is raised.

    Returns:
        The version string from `pyproject.toml` (e.g. ``"1.2.3"`` ->
        ``1.2.3``).
    """

    version = None
    with open("pyproject.toml", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("version"):
                version = line.split("=")[1].strip().strip('"')
                break
    if version is None:
        raise ValueError("Version not found in pyproject.toml")
    return version

setup()
