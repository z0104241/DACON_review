# main.py
import torch
import pandas as pd
import numpy as np
import random
import gluonnlp as nlp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from tqdm import tqdm

# 모듈 임포트
import config
from model import BERTClassifier
from dataset import BERTDataset, collate_fn
from trainer import train_model

def set_seed(seed):
    """실험 재현을 위한 시드 고정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict(model, test_dataloader, device):
    """테스트 데이터셋에 대한 예측을 수행하는 함수"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            # 레이블은 더미 값이므로 _ 로 받음
            token_ids, valid_length, segment_ids, _ = batch
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            
            outputs = model(token_ids, valid_length, segment_ids)
            # 가장 높은 확률을 가진 클래스의 인덱스를 예측값으로 사용
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    return predictions

def main():
    # 시드 고정
    set_seed(config.SEED)

    # --- 1. 토크나이저 및 모델 로드 ---
    print("Loading tokenizer and model...")
    tokenizer = KoBERTTokenizer.from_pretrained(config.MODEL_NAME)
    bert_model = BertModel.from_pretrained(config.MODEL_NAME, return_dict=False)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
    
    model = BERTClassifier(bert_model, dr_rate=config.DROPOUT_RATE).to(config.DEVICE)
    print("Tokenizer and model loaded.")

    # --- 2. 데이터 로드 및 전처리 ---
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(config.DATA_PATH + config.TRAIN_FILE)
    
    # 레이블을 숫자로 변환
    label_map = {label: i for i, label in enumerate(config.CLASSES)}
    train_df['Rating_Category'] = train_df['Rating_Category'].map(label_map)
    
    # 데이터 리스트 생성
    data_list = [[row['Reviews'], row['Rating_Category']] for _, row in train_df.iterrows()]
    
    # 학습/검증 데이터 분리
    dataset_train, dataset_val = train_test_split(data_list, test_size=0.1, random_state=config.SEED, stratify=train_df['Rating_Category'])
    print(f"Train data size: {len(dataset_train)}, Validation data size: {len(dataset_val)}")

    # --- 3. 클래스 가중치 계산 ---
    labels = train_df['Rating_Category'].values
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.FloatTensor(class_weights)
    print(f"Class weights: {class_weights}")

    # --- 4. 데이터셋 및 데이터로더 생성 ---
    train_dataset = BERTDataset(dataset_train, 0, 1, tokenizer.tokenize, vocab, config.MAX_LEN, pad=False, pair=False)
    val_dataset = BERTDataset(dataset_val, 0, 1, tokenizer.tokenize, vocab, config.MAX_LEN, pad=False, pair=False)
    
    # collate_fn에 정의한 동적 패딩 함수를 전달
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    print("DataLoader created.")

    # --- 5. 모델 학습 시작 ---
    print("Starting training...")
    train_model(model, train_dataloader, val_dataloader, class_weights)
    
    # --- 6. 예측 및 제출 파일 생성 ---
    print("\nStarting prediction...")
    # 저장된 최고 성능의 모델 가중치 로드
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(config.DATA_PATH + config.TEST_FILE)
    # 테스트 데이터는 레이블이 없으므로 더미 값(0)으로 설정
    test_data_list = [[review, 0] for review in test_df['Reviews']]

    # 테스트 데이터셋 및 데이터로더 생성
    test_dataset = BERTDataset(test_data_list, 0, 1, tokenizer.tokenize, vocab, config.MAX_LEN, pad=False, pair=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    # 예측 실행
    predictions = predict(model, test_dataloader, config.DEVICE)

    # 예측 결과를 다시 텍스트 레이블로 변환
    idx_to_label = {i: label for label, i in label_map.items()}
    test_df['Rating_Category'] = [idx_to_label[p] for p in predictions]

    # 제출 파일 생성
    submission_df = test_df[['ID', 'Rating_Category']]
    submission_df.to_csv(config.DATA_PATH + config.SUBMISSION_FILE, index=False, encoding='utf-8-sig')
    print(f"Submission file created at: {config.DATA_PATH + config.SUBMISSION_FILE}")


if __name__ == '__main__':
    main()
