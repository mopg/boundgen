import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boundgen-mopg",
    version="0.0.1",
    author="Max Opgenoord",
    author_email="11822896+mopg@users.noreply.github.com",
    description="Algorithm to plan route over a course while simultaneously detecting cones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mopg/boundgen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)