# ElementBender

Choose an element (fire, water, earth or air) and bend it with your hands.

The name is inspired by the benders of *Avatar: The Last Airbender*.

[![Download for Windows](https://img.shields.io/badge/Windows-Download_.exe-e63946?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/ppdrada/ElementBender/releases/latest/download/ElementBender.exe)

On Windows, download the `.exe` and open it: nothing to install, you only need a webcam (or a mouse).

## Built with

- **Python 3.11+**
- **OpenCV**: opens the webcam and shows the image
- **MediaPipe**: finds your hands in the image
- **NumPy**: draws the ASCII elements

## Installation

Create a virtual environment and install the libraries:

**Windows 11**

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**macOS and Linux**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

**Windows 11**

```powershell
.venv\Scripts\python main.py
```

**macOS and Linux**

```bash
.venv/bin/python main.py
```

Stand in front of the webcam. You can also play with the mouse.

## Controls

**Menu**: show 1, 2, 3 or 4 fingers (or hover an element, or hold the keys 1-4) until it fills the screen. Click to enter right away.

**Bending**

| Hand | Mouse | Fire | Water | Earth | Air |
|---|---|---|---|---|---|
| Move | Move | Flame and trail | Water whip | Floating rock | Wind whip |
| Fist | Hold left button | Charge a fireball | Gather an orb | Raise a wall | Gather an air ball |
| Quick swing | Quick drag | Throw | Throw | Throw | Throw |
| Five fingers 🖐️ | Hold right button | The screen catches fire | Ice | Lava | Gust |

**Back to the menu**: rest your hand on the arrow at the top left, click it, or press Esc. Press Q to quit.
