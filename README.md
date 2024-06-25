# albumentations_examples

Install requitements:

```bash
pip install -r src/text_overlay/requirements.txt
```

```bash
pip install git+https://github.com/albumentations-team/albumentations.git
```

To run text overlay:

```bash
(pipelines) ➜  pipelines git:(main) ✗ python -m src.text_overlay.dataset -h
usage: dataset.py [-h] -i INPUT_PATH -f FONT_PATH -o OUTPUT_PATH

Process TAR files to extract images and metadata.

options:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path to images and documentation
  -f FONT_PATH, --font_path FONT_PATH
                        Path to font.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the output folder to save extracted images and metadata.
```
