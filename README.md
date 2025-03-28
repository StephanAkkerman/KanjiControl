# KanjiControl

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->
![banner](img/banner.png)

---
<!-- Adjust the link of the second badge to your own repo -->
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/github/license/StephanAkkerman/KanjiControl.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

## Introduction

KanjiControl is a Python project that lets you create mnemonic images of given Japanse kanji and Mandarin Hanzi characters. It's simple to use, simply run `src/main.py` and give a character and its meaning. The program will then generate a mnemonic image for you, saved in the `output` folder.

It uses the QR Code Control Net by Monster-Labs to control the Stable Difussion 1.5 model.

### Examples
This is an example of a generated mnemonic image for the kanji character "木" (tree):
![木](output/木.png)

## Table of Contents 🗂

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Installation ⚙️
<!-- Adjust the link of the second command to your own repo -->

The required packages to run this code can be found in the requirements.txt file. To run this file, execute the following code block after cloning the repository:

```bash
pip install -r requirements.txt
```

or

```bash
pip install git+https://github.com/StephanAkkerman/KanjiControl.git
```

## Usage ⌨️

## Citation ✍️
<!-- Be sure to adjust everything here so it matches your name and repo -->
If you use this project in your research, please cite as follows:

```bibtex
@misc{project_name,
  author  = {Stephan Akkerman},
  title   = {KanjiControl},
  year    = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/KanjiControl}}
}
```

## Contributing 🛠
<!-- Be sure to adjust the repo name here for both the URL and GitHub link -->
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.\
![https://github.com/StephanAkkerman/KanjiControl/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/KanjiControl)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
