import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="experiment",
    version="0.0.1",
    author="Mario Belledonne",
    author_email="mbelledonne@gmail.com",
    description="Human Galileo experiment module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages = ['experiment'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
