"""Launch the 2D motion-analysis window.

Run from the project folder:

    python app.py

Close RealSense Viewer before Live. This is 2D only (camera pixels).
"""

from src.ui.app import main


if __name__ == "__main__":
    main()
