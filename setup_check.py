"""
Environment check for Adaptive Proctoring Agent.
Run: python setup_check.py
All lines should print OK.
"""

import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("gymnasium", "gymnasium"),
    ("stable_baselines3", "stable-baselines3"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]

all_ok = True
for module, pkg in REQUIRED:
    try:
        m = __import__(module)
        version = getattr(m, "__version__", "unknown")
        print(f"  OK  {pkg:<22} ({version})")
    except ImportError:
        print(f"  MISSING  {pkg}  →  pip install {pkg}")
        all_ok = False

print()
print(f"Python {sys.version}")
print()
if all_ok:
    print("All dependencies present. Environment is ready.")
else:
    print("Fix missing packages above, then re-run.")
    sys.exit(1)
