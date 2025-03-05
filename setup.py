import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "kidney-Disease-Classification-Deep-learning-Project"
AUTHER_USER_NAME = "masumcse22"
SRC_REPO = "cnnClassifier"
AUTHER_EMAIL = "masum.cse2022@gmail.com"



setuptools.setup(
    name = SRC_REPO,
    version=__version__,
    author=AUTHER_USER_NAME,
    author_email=AUTHER_EMAIL,
    description="A small package for kidney disease classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= f"https://github.com/{AUTHER_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://githib.com/{AUTHER_USER_NAME}/{REPO_NAME}/issues",
    },

    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)