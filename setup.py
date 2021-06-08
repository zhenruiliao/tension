import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorforce",
    version="0.0.1",
    author="Zhenrui Liao",
    author_email="zhenrui.liao@columbia.edu",
    description="A Python package for training chaotic RNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhenruiliao/TensorFORCE",
    project_urls={
        "Bug Tracker": "https://github.com/zhenruiliao/TensorFORCE/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

