from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="SplitFVM",
    version="0.2",
    description="1D Finite-Volume Split Newton Solver",
    url="https://github.com/gpavanb1/SplitFVM",
    author="gpavanb1",
    author_email="gpavanb@gmail.com",
    license="MIT",
    packages=["splitfvm", "splitfvm.equations"],
    install_requires=["numpy", "numdifftools", "matplotlib", "splitnewton"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="amr newton python finite-volume armijo optimization pseudotransient splitting",
    project_urls={  # Optional
        "Bug Reports": "https://github.com/gpavanb1/SplitFVM/issues",
        "Source": "https://github.com/gpavanb1/SplitFVM/",
    },
    zip_safe=False,
)
