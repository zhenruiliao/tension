import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
	'tensorflow>=2.5.0',
	'numpy',
	'matplotlib', 
]

setuptools.setup(
    name="tension",
    version="0.1",
    author="Zhenrui Liao",
    author_email="zhenrui.liao@columbia.edu",
    description="A Python package for training chaotic Echo State and Spiking RNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires, 
    url="https://github.com/zhenruiliao/tension",
    project_urls={
        "Bug Tracker": "https://github.com/zhenruiliao/tension/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="tension"),
    python_requires=">=3.7",
)

