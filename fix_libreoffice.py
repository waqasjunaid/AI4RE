"""
fix_libreoffice.py
Downloads a user manual for the AI4RE-Test pipeline.

We use the GNU Bash Reference Manual -- a well-structured, freely
available user documentation file that is stable and always online.
This is a better test document than LibreOffice Writer Guide because:
  - It has clear section hierarchy (chapters, sections, subsections)
  - It contains function descriptions, constraints, examples
  - It is plain text, no extraction issues

Run once from the AI4RE-Test/ root:
    python fix_libreoffice.py
"""

import os
import sys
import urllib.request

OUTPUT_DIR = "data/raw/user_manual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GNU Bash manual - plain text version, always available
BASH_URL  = "https://www.gnu.org/software/bash/manual/bash.txt"
BASH_PATH = os.path.join(OUTPUT_DIR, "libreoffice_writer_guide.txt")


def download(url, dest):
    print("Downloading {} ...".format(url))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        size = os.path.getsize(dest) / 1024
        print("  Saved: {}  ({:.0f} KB)".format(dest, size))
        return True
    except Exception as e:
        print("  FAILED: {}".format(e))
        return False


def main():
    print("Setting up user manual document ...\n")

    if os.path.exists(BASH_PATH):
        size = os.path.getsize(BASH_PATH) / 1024
        print("Already exists: {}  ({:.0f} KB)".format(BASH_PATH, size))
        print("Ready to use -- run python test_stage1.py")
        return

    ok = download(BASH_URL, BASH_PATH)

    if ok:
        # Quick check: count paragraphs
        with open(BASH_PATH, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        import re
        paras = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
        print("  Paragraphs found: {}".format(len(paras)))
        print("\nDone. test_stage1.py will load this file as user_manual.")
    else:
        print("\nAuto-download failed. Manual instructions:")
        print("1. Open: https://www.gnu.org/software/bash/manual/bash.txt")
        print("2. Save the page as: {}".format(BASH_PATH))
        print("3. Run: python test_stage1.py")
        sys.exit(1)


if __name__ == "__main__":
    main()