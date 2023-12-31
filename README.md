## 🤖 Tensorfloom

- Deep learning web project  

- Hosted on TBD

- MVP Deep Learning models deployed

## 📂 Code-base structure

The project is coded using a simple and intuitive structure presented below:

```bash
< PROJECT ROOT >
   |
   |-- pre_trained_model/                # Directory to hold pre-trained model
   |    |-- assets/                      # Assets directory 
   |    |    |-- tokens.txt              # Tokens for words file         
   |    |
   |    |-- variables/                   # Variables file        
   |    |-- saved_model.pb               # Deep learning model
   |
   |-- static/                           # Static directory
   |    |-- css/                         # CSS directory                     
   |         |-- main.css                # Main CSS file  
   |
   |-- templates/                        # Templates directory
   |    |-- index.html                   # index html
   |
   |-- app.py                            # Deep Learning word analysis
   |-- cake.py                           # Deep Learning food recognition
   |-- test.py                           # Tests Deep Learning function
   |-- utils.py                          # Calls pretrained model
   |-- README.md                         # Standard readme documentation
   |-- requirements.txt                  # Required libraries file
   |
   |-- ************************************************************************
```

<br />

## 📚 Libraries Used

- `Tensorflow` - Deep Learning Library
- `Flask` - Basic Web Framework used for backend
- `OpenCV` - Vision and image working Library
- `Pandas` - Reading and working with CSV files
- `Numpy` - Working with tensors and numpy arrays

## 🤖 텐서프룸

> 딥러닝 프로젝트
>
> 라이브 사이트
>
> MVP 신경망 프로젝트

## 📂 코드 기반 구조

이 프로젝트는 아래에 제시된 간단하고 직관적인 구조를 사용하여 코딩됩니다.

```bash
<프로젝트 루트>
    |
    |-- pre_trained_model/        # 사전 훈련된 모델을 저장하는 디렉토리
    |   |-- assets/               # 자산들을 저장하는 디렉토리
    |   |   |-- tokens.txt        # 단어들의 토큰들을 저장하는 파일
    |   |
    |   |-- variables/            # 변수들을 저장하는 파일들
    |   |-- saved_model.pb        # 딥 러닝 모델 파일
    |
    |-- static/                   # 정적 파일들을 저장하는 디렉토리
    |   |-- css/                  # CSS 파일들을 저장하는 디렉토리
    |   |-- main.css              # 주 CSS 파일
    |
    |-- templates/                # 템플릿 파일들을 저장하는 디렉토리
    |   |-- index.html            # 인덱스 HTML 파일
    |
    |-- app.py                    # 딥 러닝 단어 분석을 위한 파일
    |-- cake.py                   # 딥 러닝 음식 인식을 위한 파일
    |-- test.py                   # 딥 러닝 기능을 테스트하기 위한 파일
    |-- utils.py                  # 사전 훈련된 모델을 호출하는 파일
    |-- README.md                 # 표준 readme 문서
    |-- requirements.txt          # 필요한 라이브러리들을 명시한 파일
    |
    |-- ************************************************************************
```

<br />

## 📚 사용된 라이브러리

- `Tensorflow` - 딥러닝 라이브러리
- `Flask` - 백엔드에 사용되는 기본 웹 프레임워크
- `OpenCV` - 이미지 파일 작업 및 처리
- `Pandas` - CSV 파일 읽기 및 작업
- `Numpy` - 텐서 및 numpy 배열 작업
