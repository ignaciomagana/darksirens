from setuptools import setup, find_packages

__minimum_jax_version__ = '0.4.34'

setup_requires = ['jax>=' + __minimum_jax_version__]

with open("requirements.txt", "r") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("README.md", "r") as fh:
    long_description = fh.read()    
    
setup(
    name="darksirens",
    version='0.0.1',
    author="Ignacio Magana Hernandez",
    author_email="imhernan@andrew.cmu.edu",
    description="A package for joint gravitational wave inference with large scale galaxy surveys.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ignaciomagana/darksirens",
    setup_requires=setup_requires,
    install_requires=install_requires,

    packages=find_packages(include=["darksirens", "darksirens.*"]),
    entry_points={
        "console_scripts": [
            "darksirens_inference=darksirens.tool.darksirens_inference:main",
            "darksirens_analyze=darksirens.tool.darksirens_analyze:main",
            "darksirens_pixelate=darksirens.tool.darksirens_pixelate:main",
            "darksirens_inference_lensing=darksirens.tool.darksirens_inference_lensing:main",
        ]
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)