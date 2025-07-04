import sys
import os
from ctypes import CDLL, c_void_p
import sysconfig
print('Paths:', sysconfig.get_paths())
try:
    lib_path = 'C:\\Users\\User\\Desktop\\tts_sdk\\exp1\\env\\vits\\Lib\\site-packages\\torchaudio\\lib\\libtorchaudio.pyd'
    print('Checking if file exists')
    if os.path.exists(lib_path):
        print('File exists, attempting to load with ctypes')
        try:
            CDLL(lib_path)
            print('Loaded successfully with ctypes')
        except OSError as e:
            print(f'Failed to load DLL: {e}. This could be due to missing dependencies like Visual C++ Redistributable (install from https://aka.ms/vs/17/release/vc_redist.x64.exe) or other system libraries.')
    else:
        print('libtorchaudio.pyd not found at expected path')
    import torchaudio
    print('Import successful')
except ImportError as e:
    print('Error importing torchaudio:', e)
