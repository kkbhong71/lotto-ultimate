# -*- coding: utf-8 -*-
"""엔진 적재 헬퍼 — 저장소 어디에서 실행해도 경로가 맞도록 한다."""
import importlib.util, sys, io, contextlib, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod

def load_engine(patch_backtest=True):
    """(v12, p13) 반환. v1.3 패치가 적용된 상태."""
    v12 = _load('lotto_v12', os.path.join(ROOT, 'engine', 'lotto_integrated_v1_2.py'))
    p13 = _load('lotto_v13', os.path.join(ROOT, 'engine', 'lotto_v13_patch.py'))
    p13.install(v12)
    if patch_backtest:
        p13.patch_backtest_engine(v12)
    return v12, p13

def data(name):
    return os.path.join(ROOT, 'data', name)
