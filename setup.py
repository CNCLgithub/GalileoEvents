import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="galileo_ramp",
    version="0.1.0",
    author="Mario Belledonne",
    author_email="mbelledonne@gmail.com",
    description="Human Galileo experiment module for the ball ramp scenario",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages = ['galileo_ramp'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
