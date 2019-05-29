---
layout: "post"
title: "CUDA intstall/uninstall"
author: "HJ-harry"
mathjax: true
---

## Why this post exists

한 대학원 수업의 파이널 팀 프로젝트를 해야 해서 다른 사람이 짠 코드를 돌릴 일이 있었는데,
디버깅을 완전히 해서 이상이 없는 것을 확인했는데도 ```cuda runtime error (11)```이 떴다.
한참을 씨름하다가 결국 알게 된 사실이 이게 결국은 **CUDA 버전 문제** 였다는 건데, 사실 아직까지
pytorch나 CUDA 버전 갖고 큰 애를 먹은 적이 없기에 해결하는데 엄청난 애를 먹었다.
내가 사용하고 있던 버전들은 다음과 같았다.

- Ubuntu 16.04, pytorch 0.4.1, CUDA 7.5

이 때까지는 내가 짠 코드를 돌리는 데에 큰 무리가 없었지만, 사실 위 버전들이 꽤 구(?)버전들이고,
언젠가는 바꿨어야 할테니 나중에 할 고생을 미리 했다고 생각하기로 했다. 그리고 앞으로 이렇게 고생을
한 문제가 있다면, 내가 보기 위해 블로그에 글로 올릴 예정이다. 결론적으로 업데이트 한 버전들은 다음과 같다.

- Ubuntu 18.04, pytorch 1.1.0, CUDA 10.0

## Ubuntu re-install

우분투를 새로 설치하기 위해서 먼저 **usb 드라이브에 설치 파일을 받았다.**

[https://www.ubuntu.com/download/desktop](https://www.ubuntu.com/download/desktop)

이 사이트에 들어가면 .iso 확장자의 파일을 받을 수 있고, 이게 나중에 부팅용으로 이용된다.
다음으로,

[http://rufus.akeo.ie/](http://rufus.akeo.ie/)

이 사이트에 들어가서 **Rufus 3.1 portable** 을 받으면, usb를 설치 디스크로 만드는 작업을
수행할 수 있다. 재부팅하면서 usb를 꽂으면 ubuntu를 새로 install할 수 있다.

## CUDA re-install

사실 CUDA만 쓰던거 잘 지우고 다시 새 거 깔았으면 위 step을 할 필요도 없었는데, 이게 버전이
꼬이면서 이것저것 잘 못 받고 잘 못 지우고를 반복하다보니 도저히 되돌릴 수 없는 경지에 도달했다. 그래서 결국 선택한 것이 포맷인데, 이 파트를 다시 할 일이 있으면 잘 해서 우분투를 포맷하는 일은 없도록 하자.

**가장 중요한 건 원래 쓰던 CUDA를 EXHAUSTIVE하게 설정까지 모두 지우는 일이다.**

[https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu](https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu)

솔루션은 위 페이지에서 거의 그대로 가져왔는데, 구글에는 여러가지 솔루션이 올라와있지만
그 중에 어떤게 나한테 될지는 아무도 모르는 일 같다.

먼저, CUDA PPA를 모두 지워야 한다. 작은 설정들까지 모두 지우는 것을 의미한다.
있다면 ```nvidia-cuda-toolkit``` 까지 모두 삭제한다.

```
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove nvidia-cuda-toolkit
```

새로운 드라이버를 설치하기 전 nvidia driver도 일단 지우는게 좋다. 드라이버 버전이랑
CUDA 버전은 같이 갈 수밖에 없는 문제이고, 서로 호환되는 버전이 따로 있다.

```
sudo apt remove nvidia-*
```

여기까지 됐으면 시스템을 업데이트 한다.

```
sudo apt update
```

다음 스텝들은 순차적으로 하면 되는데, NVIDIA 사이트로부터 무엇을 받을지 설정하는 부분이다.

```
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install cuda-10-0
```

여기까지 하면 CUDA 10.0이 깔리는게 맞다. 이와 함께 NVIDIA 드라이버 418.40이 깔리게 된다.

**cuDNN도 깔아버리자 (버전 7 7.5.0.56)**

```
sudo apt install libcudnn7
```

``` ~./profile ``` 에 다음과 같은 라인을 추가해주자.

```
# set PATH for cuda 10.0 installation
if [ -d "/usr/local/cuda-10.0/bin/" ]; then
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

여기까지 했으면 **컴퓨터를 재부팅해서 설정을 잘 맞춰주자**

1. NVIDIA version check: ```nvcc --version```
2. cuDNN version check:
```
/sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
```
그러면 output으로
```
terrance@terrance-ubuntu:~$ /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
    libcudnn.so.7 -> libcudnn.so.7.5.0
```

가 나오면 성공.

3. nvidia-smi check

```
terrance@terrance-ubuntu:~$ nvidia-smi
Sat Mar 23 20:52:18 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.40       Driver Version: 418.40       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 760     Off  | 00000000:02:00.0 N/A |                  N/A |
| 43%   41C    P8    N/A /  N/A |    122MiB /  1998MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0                    Not Supported                                       |
+-----------------------------------------------------------------------------+
```
이게 나오면 성공이다.
