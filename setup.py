from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
# buildOptions = dict(packages = ["h5py",
#                         "Jinja2", "Keras-Applications", "Keras-Preprocessing", "Markdown",
#                         "MarkupSafe", "mock", "numpy", "onnx", "onnx-tf", "onnxruntime",
#                         "opencv-python", "pefile", "protobuf", "pywin32-ctypes", "PyYAML", "six",
#                         "tensorboard", "tensorflow", "tensorflow-estimator", "termcolor", "typing-extensions", "Werkzeug"], excludes = [])

import sys
base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('source.py')
]

setup(name='face distance',
      version = '0.1',
      description = '',
    #   options = dict(build_exe = buildOptions),
      executables = executables)
