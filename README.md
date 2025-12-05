# MLLM-based Video Temporal Grounding (UniTime & Time-R1)

본 저장소는 **UniTime**과 **Time-R1**을 동일한 베이스 모델(**QwenVL)로 재현·비교하기 위한 코드/스크립트를 제공합니다.  
모든 실험은 **vision encoder/LLM 동결 + LoRA (rank=32, alpha=32)**로 수행하며, 입력 영상은 **2 fps**로 샘플링합니다.  
벤치마크: **Charades-STA**.

---

## 1) 환경 (Environment)

- **Python**: 3.10 권장
- **PyTorch**: CUDA 12.x 예시
- **필수 패키지**: `transformers`, `accelerate`, `decord`, `peft`, `deepspeed`, `flash-attn` 등

```bash
conda create -n vtg python=3.10 -y
conda activate vtg

각 코드의 requirements.txt필요한 libiary 

pip install -r requirements.txt
```

## 2) 훈련 (Training)

- **UniTime**

  - 루트 이동 후 학습 스크립트 실행

  ```bash
  cd UniTime
  bash scripts/train.sh
  ```

- **Time-r1**

  - 루트 이동 후 학습 스크립트 실행

  ```bash
  cd time-r1
  # first preprocess dataset
  bash scripts/finetune/preprocess_videos_ch.sh
  # then finetune
  bash scripts/finetune/run_charades.sh
  ```



## 3) 평가 (Evaluation)

- **UniTime**

  - 기본 평가 스크립트 및 지표 산출

  ```
  cd UniTime
  bash scripts/eval.sh
  python eval_metrics.py --res ../UniTime/results/charades/baseline.json
  ```

- **Time-R1**

- 테스트 스크립트 실행

```
cd time-r1
bash scripts/test.sh
```

