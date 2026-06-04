"""
patch_coverage.py
-----------------
Fixes the keyword length threshold bug in the coverage checker.

The bug: words <= 4 chars were filtered out, causing items like
"Find a pet by ID" and "Get user by user name" to produce zero
keywords and never match any TR.

Fix: lower threshold from > 4 to >= 3.

Run: python patch_coverage.py
"""
import os, py_compile

TARGET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "evaluation", "tr_reviewer.py"
)

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

OLD = "if len(w) > 4"
NEW = "if len(w) >= 3"

count = content.count(OLD)
if count == 0:
    print("Pattern '{}' not found -- may already be patched.".format(OLD))
else:
    content = content.replace(OLD, NEW)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(content)

    bad = [b for b in open(TARGET, "rb").read() if b > 127]
    try:
        py_compile.compile(TARGET, doraise=True)
        syn = "OK"
    except Exception as e:
        syn = str(e)

    print("Fixed {} occurrence(s) in:".format(count))
    print("  {}".format(TARGET))
    print("non-ASCII: {}  syntax: {}".format(len(bad), syn))
    print()
    print("Now run:  python test_stage5_review.py")
