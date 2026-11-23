"""세 백엔드가 모두 GPU를 인식하는지 확인한다. 하나라도 실패하면 종료 코드 1."""

import subprocess
import sys

CHECKS = {
    "jax": (
        "import jax; d = jax.devices(); print(jax.__version__, d); "
        "assert d[0].platform == 'gpu'"
    ),
    "tensorflow": (
        "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); "
        "print(tf.__version__, g); assert g"
    ),
    "torch": (
        "import torch; print(torch.__version__, torch.version.cuda); "
        "assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    ),
}

failed = []
for name, code in CHECKS.items():
    # 프레임워크끼리 CUDA 라이브러리가 섞이지 않도록 각각 별도 프로세스에서 확인
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok = result.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {result.stdout.strip()}")
    if not ok:
        failed.append(name)
        print(result.stderr.strip()[-2000:])

if failed:
    sys.exit(f"GPU를 사용할 수 없는 백엔드: {', '.join(failed)}")
print("모든 백엔드에서 GPU 사용 가능")
