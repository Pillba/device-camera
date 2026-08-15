# OCR and Medication Information Extraction

This program is designed to extract medication information from images using Optical Character Recognition (OCR). It supports various fields such as medicine name, instructions, and quantity. The program also includes a feature to save intermediate images for debugging purposes.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/ocr-medication-extraction.git
   cd ocr-medication-extraction
   ```

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To run the program, use the following command:

```sh
python pill_bottle_ocr_1.py [options]
```

### Options

- `--debug-images`: Save intermediate images for debugging purposes.

### Example

To run the program and save intermediate images for debugging:

```sh
python pill_bottle_ocr_1.py --debug-images
```

## Features

- **OCR Extraction**: Extracts medicine name, instructions, and quantity from images.
- **Intermediate Image Debugging**: Saves intermediate images (gray, resized, blurred, thresholded) to the `./debug` folder when the `--debug-images` option is used.
- **Error Handling**: Handles file not found and image read errors gracefully.

## Requirements

- Python 3.8 or higher
- OpenCV (`cv2`)
- PyTesseract (`pytesseract`)
- spaCy (optional, for enhanced text extraction)

## Contributing

Contributions are welcome! Please follow the guidelines in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the OpenCV and PyTesseract communities for their excellent libraries.
- Special thanks to the contributors who have made this project possible.<|im_end|>
