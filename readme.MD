# Software requirements :

- Python

# Python Library requirements :

- scipy
- numpy
- matplotlib
- sklearn
- tensorflow
- os
- ipykernel
- tqdm
- pandas
- cv2

# File structure

## Dataset :

- dataset.ipynb : Generate a dataset of signal data that is channel coded using convolutional code.
- channel.py : Implements of channel noise effect
- conv.py : Implements of convolutional code
- seq2img.ipynb : Convert sequence data to img data

## Model :

- Recognition.ipynb : Simulation code targeting bitstream or bitstring data composed of 0s and 1s.
- Recognition_IMG.ipynb : Simulation code targeting 128x1 image data that made from a bitstring.
- Recognition_IMG_2.ipynb : Same as 1, but the target(Input) image size is 128x128.
- ...

# Approach

## Step 1: Run dataset.ipynb

## Step 2: Run seq2img.ipynb (Skip this step if you want to use the sequence dataset)

```
- If you want image dataset that size is 128x1, Run Cell 1
- If you want image dataset that size is 128x128, Run Cell 2

```

## Step 3: Run the model code

```
- If you skipped Step 2, run Recognition.ipynb
- If not, run Recognition_IMG.ipynb or Recognition_IMG_2.ipynb

```

---

# 파일 실행방법

IDE (IntelliJ, VS Code, Visual Studio, Jypyter Notebook 등)에서 ipynb 파일을 열어 순차적으로 Cell을 실행
py 파일은 클래스 정의 파일로 실행할 필요 없음

데이터 셋 생성은 dataset.ipynb 에서 Cell 1 - Cell 2 만 실행

- Cell 2에서 Line 1은 데이터 셋이 생성되는 위치를 지정함 : 절대경로로 작성
- 현재는 Convolutional code만 생성하여 사용하였기 때문에 Cell 2 까지만 실행

데이터 셋 변환은 seq2img.ipynb 에서 Cell 1 - Cell 2 실행

- 각 Cell 의 base 와 dest 변수 값을 지정하여 변환할 데이터 셋의 경로와 변환된 데이터 셋의 경로를 설정
- Cell 1 은 128X1 크기의 이미지 파일을 생성
- Cell 2 는 128X128 크기의 이미지 파일을 생성

모델 실행은 각 Recognition.ipynb 파일에서 실행
Import Statements 부터 순차적으로 Cell 을 실행

- Cell 6에서 CH_CODE 변수 값을 사용, 채널코딩 종류를 설정하여 분류할 클래스를 지정
- Cell 7 ~ 에서 데이터 셋을 불러옴
- 이후 불러온 데이터 셋을 Markdown을 따라 모델을 선택하여 Compile , Fit 수행
- 학습된 모델은 손실 추이, 클래스 별 예측 확률, 혼동행렬, SNR 별 분류 정확도를 확인할 수 있음
