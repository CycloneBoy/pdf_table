import os

from setuptools import setup

description = "PdfTable: pdf table extract tool"

VERSION = '0.0.1'

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()


def read(file: str):
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


def read_readme():
    return read("README.md")


def read_requirements():
    content = read("requirements.txt")
    packages = content.split("\n")
    return packages


def main():
    setup(
        name="pdftable",
        package_dir={"": "src"},
        version=VERSION,
        author="CycloneBoy",
        author_email="xuanfeng1992@gmail.com",
        description=description,
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/CycloneBoy/pdf_table",
        keywords=["PDF", "OCR", "Table Extraction", "Document Intelligence"],
        install_requires=REQUIRED_PACKAGES,
        python_requires=">=3.6",
        entry_points={"console_scripts": ["pdftable=pdftable.cli:main"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            'Topic :: Text Processing :: Linguistic',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        license="Apache 2.0",
    )


if __name__ == '__main__':
    main()
