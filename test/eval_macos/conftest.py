"""Make ``harness`` importable from every suite in this directory.

``pytest.ini`` sets ``pythonpath = . python``, which covers the committed suites but
not this one, so without this the suites would each need a ``sys.path`` prelude.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
