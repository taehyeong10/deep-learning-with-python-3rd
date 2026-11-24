# 2장 실행 가이드

`chapter02_mathematical-building-blocks.ipynb`를 실행하기 전에 준비할 것을 정리한 문서입니다.
실행 환경마다 다른 값(백엔드, 데이터 위치)은 노트북에 직접 쓰지 않고 `.env` 파일에 둡니다.

## 1. 설정 파일 준비

`.env.example`을 `.env`로 복사한 뒤 값을 환경에 맞게 고칩니다.

```bash
cp .env.example .env
```

| 변수 | 설명 | 2장에서의 값 |
|---|---|---|
| `KERAS_BACKEND` | 케라스 백엔드 (`tensorflow` / `jax` / `torch`) | `tensorflow` |
| `DATASET_DIR` | 데이터셋 파일을 모아 둔 디렉터리. 노트북이 여기서 파일을 직접 읽습니다 | 환경에 맞는 절대 경로 |

- `.env`는 환경마다 값이 다르고 내부 경로가 들어가므로 저장소에 올리지 않습니다(`.gitignore`에 등록되어 있습니다). 저장소에는 예시인 `.env.example`만 둡니다.
- `.env`는 **노트북 커널이 실행되는 디렉터리**(또는 그 상위)에 있어야 합니다. 원격에서 실행한다면 원격 쪽 작업 디렉터리에 두어야 하고, 로컬에만 있으면 읽지 못합니다. IDE가 프로젝트를 원격으로 동기화하는 경우 `.env`는 함께 올라가지 않을 수 있으니 직접 복사합니다. 커널의 작업 디렉터리는 노트북에서 `os.getcwd()`로 확인할 수 있습니다.
- 2장 뒷부분의 `%%backend tensorflow` 셀은 `KERAS_BACKEND`가 `tensorflow`일 때만 실행됩니다.

## 2. 데이터셋 준비

2장에서 쓰는 데이터는 MNIST 하나입니다.

| 데이터셋 | 노트북에서의 호출 | 파일 위치 |
|---|---|---|
| MNIST | `load_mnist()` | `$DATASET_DIR/mnist.npz` |

노트북은 데이터를 인터넷에서 내려받지 않고, `DATASET_DIR`에 넣어 둔 파일을 직접 읽습니다.
설정 셀이 `.env`의 `DATASET_DIR`로 경로를 만들고 `load_mnist()`를 정의합니다. 원래 예제의 `mnist.load_data()` 자리에 이 함수를 씁니다. 반환 형태는 같습니다.

```python
MNIST_PATH = os.path.join(os.environ["DATASET_DIR"], "mnist.npz")

def load_mnist():
    with np.load(MNIST_PATH, allow_pickle=True) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])
```

실제 디렉터리 경로는 `.env`에만 있고 노트북에는 남지 않습니다.

> `mnist.load_data(path=...)`에 절대 경로를 넘기는 방식은 쓸 수 없습니다. 현재 케라스는 `Paths are no longer accepted as the fname argument` 오류를 냅니다.

### 파일 넣는 방법

실행 환경이 외부에 접속할 수 없어도 되도록, 인터넷이 되는 PC에서 파일을 받아 넣어 둡니다.

1. 인터넷이 되는 PC에서 내려받습니다.

   ```bash
   curl -fLO https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
   ```

2. 파일이 온전한지 확인합니다.

   ```bash
   shasum -a 256 mnist.npz
   # 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
   ```

3. 실행 환경의 `DATASET_DIR`로 복사합니다. 디렉터리가 없으면 먼저 만듭니다.

   ```bash
   ssh -i <키 파일> -p <포트> <사용자>@<호스트> 'mkdir -p <DATASET_DIR>'
   scp -i <키 파일> -P <포트> mnist.npz <사용자>@<호스트>:<DATASET_DIR>/
   ```

`DATASET_DIR`이 컨테이너의 임시 저장 공간에 있으면 환경을 다시 띄울 때 파일이 사라집니다. 그때는 3번을 다시 하거나, `DATASET_DIR`을 영구 볼륨 아래로 잡습니다.

## 3. 노트북 실행

1. 맨 위 설정 셀을 가장 먼저 실행합니다. 이 셀이 `.env`를 읽어 환경 변수로 등록하고 `load_mnist()`를 정의합니다. `MNIST 파일 있음: True`가 출력되는지 확인합니다. `KERAS_BACKEND`는 `keras`를 임포트하기 전에 정해져야 적용됩니다.
2. 나머지 셀을 위에서부터 차례로 실행합니다.

`KERAS_BACKEND`를 바꿨다면 커널을 재시작한 뒤 처음부터 다시 실행합니다. 이미 임포트된 케라스에는 바뀐 백엔드가 적용되지 않습니다. `DATASET_DIR`만 바꿨다면 설정 셀을 다시 실행하면 됩니다.

## 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| `.env 파일이 없습니다` | 커널이 실행되는 디렉터리(또는 그 상위)에 `.env`가 없습니다. 1번을 확인합니다 |
| `MNIST 파일 있음: False` 또는 `FileNotFoundError: … mnist.npz` | `$DATASET_DIR/mnist.npz`가 없습니다. 2번을 확인합니다 |
| `NameError: name 'load_mnist' is not defined` | 설정 셀을 실행하지 않았습니다. 맨 위 셀부터 실행합니다 |
| `이 셀은 tensorflow 백엔드가 필요합니다` | `KERAS_BACKEND`가 `tensorflow`가 아닙니다. `.env`를 고치고 커널을 재시작합니다 |
