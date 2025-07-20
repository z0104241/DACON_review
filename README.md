# 📝 KoBERT를 이용한 한국어 리뷰 감성 분석 (최종1위)
 


이 저장소는 DACON의 리뷰 데이터를 사용하여, SKT의 KoBERT 모델을 기반으로 텍스트의 감성(긍정, 중립, 부정)을 분류하는 딥러닝 모델의 전체 개발 과정을 담고 있습니다.


## 💡 핵심 전략 및 특징 (Key Features)

* 사전 학습 모델 활용 (model.py):
  `transformers` 라이브러리를 통해 `skt/kobert-base-v1` 모델을 불러온 뒤, `BERTClassifier` 클래스에서 이 모델을 기반으로 분류 레이어를 추가하여 Fine-tuning을 수행합니다. 이를 통해 강력한 한국어 이해 능력을 감성 분석 태스크에 효과적으로 전이시킵니다.

* 효율적인 학습 파이프라인:
  - 동적 패딩 (dataset.py): `DataLoader` 생성 시 `collate_fn` 함수를 인자로 전달합니다. 이 함수는 각 배치(batch) 단위로 문장들을 처리하며, 해당 배치 내에서 가장 긴 문장을 기준으로만 패딩을 적용합니다. 이는 불필요한 패딩 계산을 줄여 학습 속도와 메모리 효율성을 크게 향상시킵니다.
  - 모델 체크포인팅 (trainer.py): `train_model` 함수 내에서 매 에폭(epoch)마다 검증 데이터셋의 정확도를 측정합니다. 만약 현재 정확도가 이전의 최고 정확도(`best_val_acc`)보다 높을 경우, `torch.save()`를 호출하여 해당 시점의 모델 가중치를 `best_model.pt` 파일로 저장합니다. 이를 통해 전체 학습 과정 중 가장 성능이 좋았던 모델을 확보할 수 있습니다.

* 안정적인 학습:
  - 클래스 불균형 처리 (main.py, trainer.py): `main.py`에서 `scikit-learn`의 `compute_class_weight` 함수를 사용하여 학습 데이터의 클래스별 불균형을 계산하고 가중치를 생성합니다. 이 가중치는 `trainer.py`의 `CrossEntropyLoss` 손실 함수에 `weight` 파라미터로 전달되어, 데이터 수가 적은 클래스에 더 높은 페널티를 부여함으로써 모델이 모든 클래스를 균형 있게 학습하도록 돕습니다.
  - 학습률 스케줄러 (trainer.py): `transformers`의 `get_cosine_schedule_with_warmup`을 사용합니다. 학습 초기에는 학습률을 서서히 증가시켜(Warm-up) 불안정한 학습을 방지하고, 이후에는 코사인(Cosine) 함수 형태에 따라 점진적으로 학습률을 감소시켜 모델이 최적점에 안정적으로 수렴하도록 유도합니다.

* 모듈화된 코드:
  프로젝트의 각 기능을 독립적인 파일로 분리하여 코드의 재사용성과 유지보수성을 높였습니다. `config.py`에서 실험 설정을, `dataset.py`에서 데이터 처리를, `model.py`에서 모델 구조를, `trainer.py`에서 학습 로직을 관리하며, `main.py`는 이들을 종합하여 전체 파이프라인을 실행합니다.


## 🚀 성능 개선 과정 (Experiments & Results)


초기 베이스라인 모델부터 시작하여, 다양한 가설을 검증하는 실험을 통해 점진적으로 성능을 개선했습니다.

| 단계 | 실험 내용                 | 모델 / 방법                      | F1 Score | 비고                               |
|:----:|:--------------------------|:---------------------------------|:--------:|:-----------------------------------|
|  1   | **베이스라인** | 기본 Fine-tuning                 |  0.590   | 성능 개선의 기준점 설정            |
|  2   | 하이퍼파라미터 튜닝       | Learning Rate, Batch Size 조정   |  0.594   | 기본적인 최적화 진행               |
|  3   | 스케줄러 적용             | + Cosine Annealing Scheduler     |  0.598   | 안정적인 학습 유도                 |
|  4   | **클래스 불균형 처리** | **+ Class Weights 적용** |  **0.606** | **가장 큰 폭의 성능 향상, 핵심 전략** |
|  5   | 최종 모델 최적화          | Dropout Rate, Epochs 미세 조정   |  0.608   | 과적합 방지 및 세부 튜닝           |

* 최종 결론: 이 과업에서는 일반적인 하이퍼파라미터 튜닝보다, 데이터의 클래스 불균형 문제를 직접적으로 해결하는 **클래스 가중치 적용**이 성능 향상에 가장 결정적인 요소였음을 확인했습니다.





 ## ⚙️ 프로젝트 구조

    .
    ├── main.py             # 전체 프로세스를 실행하는 메인 스크립트
    ├── trainer.py          # 모델 학습 및 평가, 체크포인팅 로직
    ├── model.py            # PyTorch 모델 아키텍처 정의
    ├── dataset.py          # 데이터셋 클래스 및 동적 패딩 함수
    ├── config.py           # 하이퍼파라미터 및 경로 설정
    ├── requirements.txt    # 필요한 라이브러리 목록
    └── data/               # train.csv, test.csv 위치
        ├── train.csv
        └── test.csv



 ## 🛠️ 기술 스택 (Tech Stack)

* Base Model: skt/kobert-base-v1
* Core Libraries: PyTorch, transformers, kobert-tokenizer
* Data Handling: pandas, scikit-learn, gluonnlp
* Dependencies: numpy, tqdm, mxnet, sentencepiece
