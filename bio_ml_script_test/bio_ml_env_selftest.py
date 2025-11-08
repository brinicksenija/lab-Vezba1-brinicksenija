#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bio_ml_env_selftest.py
Notebook-friendly self-test for a conda environment used in computational biology + ML (CPU-only).

• Collects Python/OS/conda metadata
• Imports and version-checks key packages (NumPy/SciPy/Pandas/Matplotlib/Seaborn/
  scikit-learn/Statsmodels/NetworkX/BioPython/scikit-bio/matplotlib-venn/TensorFlow/PyTorch/Jupyter)
• Runs tiny CPU smoke tests (NumPy, scikit-learn, TensorFlow, PyTorch)
• Reports Jupyter kernel registration and BLAS/LAPACK hints
• Returns a dict report and (optionally) writes JSON + plots

Use from a notebook:
    from bio_ml_env_selftest import run_all_tests, show_report
    report = run_all_tests(outdir="bio_ml_env_test_out", plots=True, quick=False, as_notebook=True)
    show_report(report)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

# Keep TF logs quiet before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# --------------------------- small helpers ----------------------------------

def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return bool(ip and ip.config and "IPKernelApp" in ip.config)
    except Exception:
        return False

def _run_cmd(cmd) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

def _maybe_set_agg_backend():
    # Use a non-interactive backend outside notebooks, before importing pyplot
    if not _is_notebook():
        import matplotlib
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

def _import_version(modname: str):
    try:
        mod = __import__(modname)
        ver = getattr(mod, "__version__", None)
        return True, mod, ver
    except Exception as e:
        return False, None, str(e)

def _numpy_blas_info(np_mod):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            np_mod.__config__.show()
        except Exception:
            pass
    txt = buf.getvalue()
    hint = None
    for k in ("mkl", "openblas", "blis", "accelerate", "atlas"):
        if k.lower() in txt.lower():
            hint = k.upper()
            break
    return hint or "Unknown", txt


# ------------------------------ tests ---------------------------------------

def _test_numpy(report: Dict[str, Any]):
    ok, np, ver = _import_version("numpy")
    if not ok:
        report["numpy"] = {"present": False, "error": ver}
        return
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    a = rng0.random((256, 256))
    b = rng1.random((256, 256))
    s = float((a @ b).sum())
    hint, cfg = _numpy_blas_info(np)
    report["numpy"] = {"present": True, "version": ver, "dot_sum": s, "blas_hint": hint,
                       "blas_config_snippet": cfg.splitlines()[:12]}

def _test_scipy(report: Dict[str, Any]):
    ok, sp, ver = _import_version("scipy")
    if not ok:
        report["scipy"] = {"present": False, "error": ver}
        return
    import numpy as np
    M = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    w = sp.linalg.eigvalsh(M)
    report["scipy"] = {"present": True, "version": ver, "eigvals": np.round(w, 6).tolist()}

def _test_pandas(report: Dict[str, Any]):
    ok, pd, ver = _import_version("pandas")
    if not ok:
        report["pandas"] = {"present": False, "error": ver}
        return
    import numpy as np
    df = pd.DataFrame({"g": ["A","A","B","B","B"], "x":[1,2,1,2,3], "y":[10,20,30,40,50]})
    gmean = df.groupby("g")["y"].mean().to_dict()
    piv_shape = list(df.pivot_table(index="g", columns="x", values="y", aggfunc="mean").shape)
    report["pandas"] = {"present": True, "version": ver, "group_mean_y": gmean, "pivot_shape": piv_shape}

def _test_matplotlib_seaborn(report: Dict[str, Any], plots: bool, outdir: Path):
    ok_m, mpl, ver_m = _import_version("matplotlib")
    ok_s, sns, ver_s = _import_version("seaborn")
    report["matplotlib"] = {"present": ok_m, "version": ver_m if ok_m else mpl}
    report["seaborn"]    = {"present": ok_s, "version": ver_s if ok_s else sns}
    if not ok_m:
        return

    _maybe_set_agg_backend()
    import matplotlib.pyplot as plt
    import numpy as np

    if plots:
        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x)
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(x, y); ax.set_title("Matplotlib: sin(x)")
        p = outdir / "plots" / "matplotlib_line.png"
        (outdir / "plots").mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        report["matplotlib"]["plots"] = {"line": str(p)}

        if ok_s:
            import seaborn as _sns  # noqa
            import pandas as pd
            df = pd.DataFrame({"a": np.random.normal(size=400), "b": np.random.normal(loc=1.0, size=400)})
            fig2, ax2 = plt.subplots(figsize=(5,3))
            _sns.kdeplot(df["a"], ax=ax2, label="a"); _sns.kdeplot(df["b"], ax=ax2, label="b")
            ax2.legend(); ax2.set_title("Seaborn: KDE")
            p2 = outdir / "plots" / "seaborn_kde.png"
            fig2.tight_layout(); fig2.savefig(p2); plt.close(fig2)
            report["seaborn"]["plots"] = {"kde": str(p2)}

def _test_sklearn(report: Dict[str, Any], quick: bool):
    ok, skl, ver = _import_version("sklearn")
    if not ok:
        report["scikit_learn"] = {"present": False, "error": ver}
        return
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    import numpy as np
    X, y = make_classification(n_samples=(250 if quick else 600),
                               n_features=12, n_informative=6, n_redundant=2, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    cv  = cross_val_score(clf, X, y, cv=(3 if quick else 5))
    report["scikit_learn"] = {"present": True, "version": ver,
                              "logreg_test_accuracy": round(acc,4),
                              "logreg_cv_mean": round(float(np.mean(cv)),4)}

def _test_statsmodels(report: Dict[str, Any]):
    ok, sm, ver = _import_version("statsmodels")
    if not ok:
        report["statsmodels"] = {"present": False, "error": ver}
        return
    import numpy as np, statsmodels.api as smapi  # type: ignore
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200,3)); beta = np.array([1.5,-2.0,0.7])
    y = X @ beta + rng.normal(scale=0.5, size=200)
    Xc = smapi.add_constant(X)
    fit = smapi.OLS(y, Xc).fit()
    report["statsmodels"] = {"present": True, "version": ver,
                             "ols_r2": round(float(fit.rsquared),4),
                             "params": [float(v) for v in fit.params[:4]]}

def _test_networkx(report: Dict[str, Any]):
    ok, nx, ver = _import_version("networkx")
    report["networkx"] = {"present": ok, "version": ver if ok else nx}

def _test_bio_packages(report: Dict[str, Any]):
    ok_bio, bio, ver_bio = _import_version("Bio")   # biopython
    ok_skb, skb, ver_skb = _import_version("skbio")
    ok_venn, venn, ver_venn = _import_version("matplotlib_venn")
    report["biopython"]      = {"present": ok_bio, "version": ver_bio if ok_bio else bio}
    report["scikit_bio"]     = {"present": ok_skb, "version": ver_skb if ok_skb else skb}
    report["matplotlib_venn"]= {"present": ok_venn, "version": ver_venn if ok_venn else venn}

def _test_tensorflow(report: Dict[str, Any]):
    ok, tf, ver = _import_version("tensorflow")
    if not ok:
        report["tensorflow"] = {"present": False, "error": tf}
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        built_cuda = tf.test.is_built_with_cuda()
    except Exception:
        gpus, built_cuda = [], False
    # Small CPU op
    try:
        with tf.device("/CPU:0"):
            x = tf.random.uniform((128,128)); y = tf.random.uniform((128,128))
            z = tf.matmul(x, y); _ = float(tf.reduce_sum(z).numpy())
        cpu_ok = True
    except Exception as e:
        cpu_ok = f"FAILED: {e}"
    report["tensorflow"] = {"present": True, "version": ver,
                            "built_with_cuda": built_cuda,
                            "visible_gpus": [d.name for d in gpus] if gpus else [],
                            "cpu_matmul_ok": cpu_ok}

def _test_torch(report: Dict[str, Any]):
    ok, torch, ver = _import_version("torch")
    if not ok:
        report["pytorch"] = {"present": False, "error": torch}
        return
    cuda_avail = False; cuda_version = None; cudnn_avail = False; cudnn_version = None; devname = None
    try:
        cuda_avail = torch.cuda.is_available()
        cuda_version = getattr(torch.version, "cuda", None)
        if hasattr(torch.backends, "cudnn"):
            cudnn_avail = torch.backends.cudnn.is_available()
            try:
                cudnn_version = torch.backends.cudnn.version()
            except Exception:
                pass
        if cuda_avail:
            devname = torch.cuda.get_device_name(0)
    except Exception:
        pass
    # Force small CPU op
    try:
        x = torch.randn((256,256), device="cpu"); y = x @ x; _ = float(y.sum().item())
        cpu_ok = True
    except Exception as e:
        cpu_ok = f"FAILED: {e}"
    report["pytorch"] = {"present": True, "version": ver,
                         "cuda_available": cuda_avail, "cuda_runtime": cuda_version,
                         "cudnn_available": cudnn_avail, "cudnn_version": cudnn_version,
                         "cuda_device0": devname, "cpu_matmul_ok": cpu_ok}

def _test_jupyter(report: Dict[str, Any]):
    ok_k, ipyk, ver_k = _import_version("ipykernel")
    report["ipykernel"] = {"present": ok_k, "version": ver_k if ok_k else ipyk}
    # kernel registry
    try:
        from jupyter_client.kernelspec import KernelSpecManager  # type: ignore
        ksm = KernelSpecManager(); specs = ksm.get_all_specs()
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
        report["jupyter_kernel_registry"] = {
            "kernel_count": len(specs),
            "contains_active_env_name": env_name in specs if env_name else False,
            "active_env_name": env_name,
            "sample_kernel_names": sorted(list(specs.keys()))[:10],
        }
    except Exception as e:
        report["jupyter_kernel_registry"] = {"error": str(e)}


# ------------------------------ public API ----------------------------------

def run_all_tests(outdir: str | Path = "bio_ml_env_test_out",
                  plots: bool = True,
                  quick: bool = False,
                  as_notebook: bool = True) -> Dict[str, Any]:
    """
    Run all checks and return a report dict. If not in a notebook and plots=True,
    PNGs will be saved under <outdir>/plots. Also writes JSON report to <outdir>.
    """
    t0 = time.time()
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"meta": {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }}

    # conda info (best effort)
    code, out, err = _run_cmd(["conda", "info", "--json"])
    if out:
        try:
            info = json.loads(out)
            report["meta"]["conda_active_prefix_name"] = info.get("active_prefix_name")
            report["meta"]["conda_root_prefix"] = info.get("root_prefix")
            report["meta"]["channels"] = info.get("channels", [])
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _test_numpy(report)
        _test_scipy(report)
        _test_pandas(report)
        _test_matplotlib_seaborn(report, plots=plots, outdir=outdir)
        _test_sklearn(report, quick=quick)
        _test_statsmodels(report)
        _test_networkx(report)
        _test_bio_packages(report)      # may show “present: False” if not installed
        _test_tensorflow(report)         # CPU smoke test
        _test_torch(report)              # CPU smoke test
        _test_jupyter(report)

    # Summaries
    ok_list, miss_list = [], []
    for k, v in report.items():
        if k in ("meta", "summary"):
            continue
        present = v.get("present", True) if isinstance(v, dict) else True
        (ok_list if present else miss_list).append(k)
    report["summary"] = {
        "ok_packages": sorted(ok_list),
        "missing_packages": sorted(miss_list),
        "elapsed_sec": round(time.time() - t0, 2)
    }

    # Write JSON report
    try:
        (outdir / "bio_ml_env_report.json").write_text(json.dumps(report, indent=2))
    except Exception:
        pass

    return report


def show_report(report: Dict[str, Any]) -> None:
    """Pretty, compact console/Notebook summary."""
    print("=== bio_ml environment self-test ===")
    meta = report.get("meta", {})
    print(f"Python: {meta.get('python')} | Env: {meta.get('conda_env')}")
    print(f"Executable: {meta.get('executable')}")
    print(f"CWD: {meta.get('cwd')}")
    okp = ", ".join(report["summary"].get("ok_packages", [])) or "(none)"
    mis = ", ".join(report["summary"].get("missing_packages", [])) or "(none)"
    print(f"Packages present : {okp}")
    print(f"Packages missing : {mis}")
    npinfo = report.get("numpy", {})
    if isinstance(npinfo, dict) and "blas_hint" in npinfo:
        print(f"BLAS backend (hint): {npinfo['blas_hint']}")
    print(f"Elapsed (s): {report['summary'].get('elapsed_sec')}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Self-test for the 'bio_ml' conda environment.")
    p.add_argument("--outdir", default="bio_ml_env_test_out", help="Output directory for report/plots.")
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    p.add_argument("--quick", action="store_true", help="Run a lighter test suite.")
    args = p.parse_args()

    rep = run_all_tests(
        outdir=args.outdir,
        plots=not args.no_plots,
        quick=args.quick,
        as_notebook=False
    )
    show_report(rep)
