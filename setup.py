# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="encoded",
    version="0.0.0",
    install_requires=requirements,
    packages=['encoded'],
    include_package_data=True,
    description="Companion repository for encoded error mitigation experiments (arXiv:XXXX.YYYYY).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Ryan LaRose",
    author_email="rmlarose@msu.edu",
    license="Apache 2.0",
    url="https://github.com/rmlarose/encoded",
    project_urls={
        "Bug Tracker": "https://github.com/rmlarose/encoded/issues/",
        "Documentation": "https://github.com/rmlarose/encoded",
        "Source": "https://github.com/rmlarose/encoded",
    },
    python_requires=">=3.10.0",
)
