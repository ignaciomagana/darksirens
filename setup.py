from setuptools import setup, find_packages

# The jax floor lives in requirements.txt (hence install_requires) — it used to
# be declared only via setup_requires, which pip ignores for dependency
# resolution while still triggering a legacy easy_install egg fetch at build
# time (fatal for an offline/no-network `pip install --no-deps .`).

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
    install_requires=install_requires,
    extras_require=extras_require,

    packages=find_packages(include=["darksirens", "darksirens.*"]),
    entry_points={
        # The two long-running GPU drivers enter through ``console_main`` (a
        # ``run_cli`` wrapper), not ``main``: run_cli's os._exit(0) skips an
        # interpreter teardown that can block for hours in the CUDA exit
        # handlers on a shared GPU, leaving a finished run to idle until the
        # SLURM cgroup OOM-kills it.  ``python -m`` gets that from the
        # ``__main__`` guard; the console scripts need it spelled out here.
        "console_scripts": [
            "darksirens_inference=darksirens.cli.inference:console_main",
            "darksirens_analyze=darksirens.cli.analyze:main",
            "darksirens_pixelate=darksirens.cli.pixelate:main",
            "darksirens_skymaps_to_samples=darksirens.cli.skymaps_to_samples:main",
            "darksirens_build_lognormal_completion=darksirens.cli.build_lognormal_completion:main",
            "darksirens_build_joint_lognormal_completion=darksirens.cli.build_joint_lognormal_completion:main",
            "darksirens_diagnose_lognormal_completion=darksirens.cli.diagnose_lognormal_completion:main",
            "darksirens_inference_lensing=darksirens.cli.inference_lensing:console_main",
        ]
    },
    classifiers=[
      "Programming Language :: Python :: 3",
      # Must match the shipped LICENSE file (MIT); the classifier is what
      # pip/PyPI report, so a stale one mislabels every built wheel.
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)