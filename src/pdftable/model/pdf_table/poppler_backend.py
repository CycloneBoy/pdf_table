# -*- coding: utf-8 -*-

import shutil
import subprocess
import traceback

__all__ = [
    "PopplerBackend",
    "PopplerBackendV2",
]

class PopplerBackend(object):
    def convert(self, pdf_path, png_path):
        pdftopng_executable = shutil.which("pdftopng")
        if pdftopng_executable is None:
            raise OSError(
                "pdftopng is not installed. You can install it using the 'pip install pdftopng' command."
            )

        pdftopng_command = [pdftopng_executable, pdf_path, png_path]

        try:
            subprocess.check_output(
                " ".join(pdftopng_command), stderr=subprocess.STDOUT, shell=True
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(e.output)


class PopplerBackendV2(object):
    def convert(self, pdf_path, png_path):

        try:
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path)

            for index, page in enumerate(images):
                save_path = png_path if index < 1 else f"{index}_{png_path}"
                page.save(save_path, 'PNG')

        except Exception as e:
            traceback.print_exc()
            raise ValueError(e.output)
