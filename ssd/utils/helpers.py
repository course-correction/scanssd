import os
import numpy as np

def get_freer_gpu():
    free = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').read().strip().split('\n')
    memory_available = [int(f.split()[2]) for f in free]
    return int(np.argmax(memory_available))
