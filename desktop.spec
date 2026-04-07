# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

project_root = "."

datas = [
    (".env.example", "."),
    ("README.md", "."),
    ("memory", "memory"),
    ("library", "library"),
    ("templates", "templates"),
    (".streamlit", ".streamlit"),
]
binaries = []
hiddenimports = []

for package_name in [
    "streamlit",
    "webview",
    "autogen_agentchat",
    "autogen_ext",
    "autogen_core",
]:
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["src/ecm_organoid_agent/desktop.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ECM Organoid Research Desk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

app = BUNDLE(
    exe,
    name="ECM Organoid Research Desk.app",
    icon=None,
    bundle_identifier="com.houjian.ecm-organoid-research-desk",
)
