import subprocess
import sys
from pathlib import Path


def main():
    app = Path(__file__).with_name("streamlit_app.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


if __name__ == "__main__":
    main()
