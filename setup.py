from setuptools import setup, find_packages

__minimum_jax_version__ = '0.4.34'

setup_requires = ['jax>=' + __minimum_jax_version__]

with open("requirements.txt", "r") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# pytest is a development dependency, not a runtime one (library review,
# architecture A2); GP population models lazily import tinygp (gp.py), which
# is deliberately optional (architecture A3).
install_requires = [r for r in install_requires if not r.startswith("pytest")]
extras_require = {
    "test": ["pytest"],
    "gp": ["tinygp"],
    # Normalizing-flow single-event surrogates (--gw_flows_path). Lazy import
    # in darksirens/gw/flows.py keeps these optional for everyone else.
    "flows": ["flowjax>=17.1,<18", "paramax", "equinox>=0.11,<0.13"],
}

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
    extras_require=extras_require,

    packages=find_packages(include=["darksirens", "darksirens.*"]),
    entry_points={
        "console_scripts": [
            "darksirens_inference=darksirens.cli.inference:main",
            "darksirens_analyze=darksirens.cli.analyze:main",
            "darksirens_pixelate=darksirens.cli.pixelate:main",
            "darksirens_skymaps_to_samples=darksirens.cli.skymaps_to_samples:main",
            "darksirens_build_lognormal_completion=darksirens.cli.build_lognormal_completion:main",
            "darksirens_diagnose_lognormal_completion=darksirens.cli.diagnose_lognormal_completion:main",
            "darksirens_inference_lensing=darksirens.cli.inference_lensing:main",
        ]
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: Apache Software License",
      "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)