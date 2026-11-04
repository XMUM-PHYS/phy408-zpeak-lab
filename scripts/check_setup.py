"""
check_setup.py  --  run this first:   python scripts/check_setup.py
"""
import importlib
import sys

REQUIRED = {"numpy": "1.22", "scipy": "1.8", "pandas": "1.4", "matplotlib": "3.5",
            "iminuit": "2.17", "jupyter": None}

ok = True
print(f"Python {sys.version.split()[0]}", "✓" if sys.version_info >= (3, 9) else "✗ (need 3.9 or newer)")
ok &= sys.version_info >= (3, 9)
for name, minimum in REQUIRED.items():
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "")
        print(f"  {name:11s} {version:10s} ✓")
    except ImportError:
        ok = False
        print(f"  {name:11s} {'missing':10s} ✗   ->  mamba install -c conda-forge {name}")

if ok:
    sys.path.insert(0, ".")
    from zlab import take_data, clean, invariant_mass, fit_zpeak
    import tempfile, pandas as pd
    r = fit_zpeak(invariant_mass(clean(pd.read_csv(take_data(1, outdir=tempfile.mkdtemp(),
                                                             verbose=False)), verbose=False)[0]))
    print(f"\nTest fit: M_Z = {r.values['M_Z']:.2f} GeV  ->  everything works. Open Z_peak_lab.ipynb!")
else:
    print("\nSomething is missing: see the README section 'Troubleshooting'.")
    sys.exit(1)
