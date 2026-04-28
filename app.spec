# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import collect_all

# 🔥 TensorFlow completo
tf_datas, tf_binaries, tf_hiddenimports = collect_all('tensorflow')

# 🔥 Matplotlib también necesita backend
mpl_datas, mpl_binaries, mpl_hiddenimports = collect_all('matplotlib')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('modelo74.h5', '.'),
        ('img', 'img'),
    ] + tf_datas + mpl_datas,
    hiddenimports=[
        'tensorflow',
        'keras',
        'cv2',
        'PIL',
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',  # 🔥 IMPORTANTE
        'matplotlib.pyplot'
    ] + tf_hiddenimports + mpl_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow.python.keras.saving.saved_model.load_context'  # evita errores raros
    ],
    noarchive=False,
)

a.binaries += tf_binaries + mpl_binaries

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='clasificador_sarro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI sin consola
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='clasificador_sarro',
)