**10월 5일 수정 내용**
 - Aspect Category 목록 정의(총 25가지) : {'제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성', '제품 전체#디자인', '패키지/ 구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반', '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', '본품#인지도', '제품 전체#가격', '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격'}

# korean_ABSA_baseline

본 소스코드는는 '2022 국립국어원 인공 지능 언어 능력 평가'의 속성 기반 감성 분석 과제의 베이스라인 모델 및 학습과 평가를 위한 코드를 제공하고 있습니다. 자세한 코드의 설명은 'src/aspect_based_sentiment_analysis_baseline.ipynb' notebook을 확인해주세요. ipynb(src/aspect_based_sentiment_analysis_baseline.ipynb) 형태의 코드와 python(sentiment_analysis.py) 파일 모두 제공하고 있으니, 선호하는 형태의 코드를 참조하면 되고, 'src/train.sh', 'src/test.sh' 두 개의 sh 파일을 이용하면 python 코드 동작에 도움이 될것입니다.



## 데이터
sample.jsonl은 국립국어원에서 제공한 데이터의 일부분이며, 전체 데이터는 국립국어원 모두의 말뭉치에서 다운받으실 수 있습니다. https://corpus.korean.go.kr/

#### example
``` 
{"id": "nikluge-sa-2022-train-00001", "sentence_form": "둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.", "annotation": [["본품#품질", ["기어", 16, 18], "negative"]]}
{"id": "nikluge-sa-2022-train-00002", "sentence_form": "이거 뭐 삐꾸를 준 거 아냐 불안하고, 거금 투자한 게 왜 이래.. 싶어서 정이 확 떨어졌는데 산 곳 가져가서 확인하니 기어 텐션 문제라고 고장 아니래.", "annotation": [["본품#품질", ["기어 텐션", 67, 72], "negative"]]}
{"id": "nikluge-sa-2022-train-00003", "sentence_form": "간사하게도 그 이후에는 라이딩이 아주 즐거워져서 만족스럽게 탔다.", "annotation": [["제품 전체#일반", [null, 0, 0], "positive"]]}
{"id": "nikluge-sa-2022-train-00004", "sentence_form": "샥이 없는 모델이라 일반 도로에서 타면 노면의 진동 때문에 손목이 덜덜덜 떨리고 이가 부딪칠 지경인데 이마저도 며칠 타면서 익숙해지니 신경쓰이지 않게 됐다.", "annotation": [["제품 전체#일반", ["샥이 없는 모델", 0, 8], "neutral"]]}
{"id": "nikluge-sa-2022-train-00005", "sentence_form": "안장도 딱딱해서 엉덩이가 아팠는데 무시하고 타고 있다.", "annotation": [["본품#일반", ["안장", 0, 2], "negative"]]}
{"id": "nikluge-sa-2022-train-00006", "sentence_form": "지금 내 실력과 저질 체력으로는 이 정도 자전거도 되게 훌륭한 거라는..", "annotation": [["제품 전체#일반", ["자전거", 23, 26], "positive"]]}
{"id": "nikluge-sa-2022-train-00007", "sentence_form": "내장 기어 3단은 썩 좋은 물건이라 기어 변환도 부드럽고 겉에서는 기어가 보이지 않기 때문에 깔끔하다.", "annotation": [["본품#품질", ["내장 기어 3단", 0, 8], "positive"]]}
{"id": "nikluge-sa-2022-train-00008", "sentence_form": "한번 교환했는데 새로 온 UD20은 불량화소가 있고 ㅜ ㅜ ㅜ", "annotation": [["본품#품질", ["UD20", 14, 18], "negative"]]}
{"id": "nikluge-sa-2022-train-00009", "sentence_form": "전에 작동 안되었던 자막 검색 후 등록 기능이 똑같이 작동 안 된다!!!", "annotation": [["본품#품질", ["자막 검색 후 등록 기능", 11, 24], "negative"]]}
{"id": "nikluge-sa-2022-train-00010", "sentence_form": "왜 [등록]키를 만들어놓고 제대로 단어장에 등록이 되지 않는 거냐!!", "annotation": [["본품#품질", ["등록]키", 3, 7], "negative"]]}
{"id": "nikluge-sa-2022-train-00011", "sentence_form": "다른 부가 기능은 참 훌륭한데..", "annotation": [["본품#품질", ["부가 기능", 3, 8], "positive"]]}
{"id": "nikluge-sa-2022-train-00012", "sentence_form": "미치겠네.", "annotation": [["제품 전체#일반", [null, 0, 0], "negative"]]}
{"id": "nikluge-sa-2022-train-00013", "sentence_form": "아.. 진짜 기계 사겠나.", "annotation": [["제품 전체#일반", [null, 0, 0], "negative"]]}
{"id": "nikluge-sa-2022-train-00014", "sentence_form": "이번에는 사전까지..", "annotation": [["제품 전체#일반", [null, 0, 0], "negative"]]}
{"id": "nikluge-sa-2022-train-00015", "sentence_form": "이런 젠장..", "annotation": [["제품 전체#일반", [null, 0, 0], "negative"]]}
```

#### 데이터 전처리
모델을 학습하기 위한 데이터 전처리는 소스코드의 tokenize_and_align_labels(tokenizer, form, annotations, max_len) 함수와 get_dataset(raw_data, tokenizer, max_len) 함수를 참고하시면 됩니다. tokenize_and_align_labels에서 원하는 형태의 데이터 형태로 가공하고, get_dataset에서 pytorch의 DataLoader를 이용하기 위한 TensorDataset 형태로 가공합니다.


## 모델 구성
Aspect Category Detection (ACD) 모델과 Aspect Sentiment Classification (ASC) 모델을 pipeline으로 연결한 모델입니다.

xlm-roberta-base(https://huggingface.co/xlm-roberta-base)를 기반으로 학습하였습니다.

학습된 baseline 모델은 아래 링크에서 받으실 수 있습니다.

Aspect Category Detection (ACD) model link: https://drive.google.com/file/d/13KpAE_7NRGuI3JnqaORuslv_6YGE-8QS/view?usp=sharing

Aspect Sentiment Classification (ASC) link: https://drive.google.com/file/d/1PEtecZW1bWpzA06SyW7nOBMX1-v-A78M/view?usp=sharing

### Aspect Category Detection (ACD)
모델 입력형태를 \<s>sentence_form\</s>\</s>카테고리\</s>와 같이하고, 각 category별로 해당 문장에서 추출 될지/ 되지 않을지에 대한 이진 분류를 합니다.
모든 카테고리에 대해 수행한 뒤, True(0)로 판별된 category를 모으면, 해당 문장에서 나타난 카테고리가 모아집니다.

입력 예시 - 모든 category에 대해
```
<s>둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.</s></s>제품 전체#일반</s>
<s>둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.</s></s>제품 전체#가격</s>
<s>둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.</s></s>제품 전체#디자인</s>
<s>둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.</s></s>본품#품질</s>
<s>둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.</s></s>본품#디자인</s>
...
```

출력 예시 - 모든 category에 대해 0 or 1 (False or True)
```
0
0
0
1
0
...
```

### Aspect Sentiment Classification (ASC)
추출 된 category에 대해 모델 입력형태를 \<s>sentence_form\</s>\</s>카테고리\</s>와 같이하고, positive, negative, neutral (0, 1, 2)로 classification 합니다.

입력 예시 - ACD에서 추출된 category에 대해서만 입력
```
<s>둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.</s></s>본품#품질</s>
```

출력 예시 - 0, 1, 2 로 출력 (positive, negative, neutral)
```
1
```

### 평가
baseline 코드에서 제공된 평가 코드로 평가하였을때, 아래와 같이 결과가 나왔습니다.
test_sentiment_analysis() 함수를 이용하면 평가를 진행할 수 있습니다.

평가함수는 evaluation_f1(true_data, pred_data) 함수를 이용하면 되고, 입력 데이터는 아래와 같습니다. 주목할 점은 true_data는 학습데이터와 형태가 똑같고, pred_data는 annotation에 ["기어", 16, 18] 와 같은 데이터는 제외하고, category와 sentiment만을 값으로 가집니다.

모델을 이용하여 pred_data와 같은 형태의 데이터를 만들기 위한 방법은 predict_from_korean_form(tokenizer, ce_model, pc_model, data) 함수를 참고하면 됩니다. 이 함수의 경우 두 개의 모델을 pipieline으로 연결하여 입력으로부터 결과를 얻어 출력과 같은 모양으로 만들어 줍니다.

true_data
``` 
{"id": "nikluge-sa-2022-train-00001", "sentence_form": "둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.", "annotation": [["본품#품질", ["기어", 16, 18], "negative"]]}
{"id": "nikluge-sa-2022-train-00002", "sentence_form": "이거 뭐 삐꾸를 준 거 아냐 불안하고, 거금 투자한 게 왜 이래.. 싶어서 정이 확 떨어졌는데 산 곳 가져가서 확인하니 기어 텐션 문제라고 고장 아니래.", "annotation": [["본품#품질", ["기어 텐션", 67, 72], "negative"]]}
{"id": "nikluge-sa-2022-train-00003", "sentence_form": "간사하게도 그 이후에는 라이딩이 아주 즐거워져서 만족스럽게 탔다.", "annotation": [["제품 전체#일반", [null, 0, 0], "positive"]]}
```


pred_data
```
{"id": "nikluge-sa-2022-train-00001", "sentence_form": "둘쨋날은 미친듯이 밟아봤더니 기어가 헛돌면서 틱틱 소리가 나서 경악.", "annotation": [["본품#품질", "negative"]]}
{"id": "nikluge-sa-2022-train-00002", "sentence_form": "이거 뭐 삐꾸를 준 거 아냐 불안하고, 거금 투자한 게 왜 이래.. 싶어서 정이 확 떨어졌는데 산 곳 가져가서 확인하니 기어 텐션 문제라고 고장 아니래.", "annotation": [["본품#품질", "negative"]]}
{"id": "nikluge-sa-2022-train-00003", "sentence_form": "간사하게도 그 이후에는 라이딩이 아주 즐거워져서 만족스럽게 탔다.", "annotation": [["제품 전체#일반", "positive"]]}
```

evaluation_t1은 다음과 같은 출력값을 가집니다.
```
{
  'category extraction result': {'Precision': 0.629059829059829, 'Recall': 0.4681933842239186, 'F1': 0.536834427425237}, 
  'entire pipeline result': {'Precision': 0.5836177474402731, 'Recall': 0.4351145038167939, 'F1': 0.49854227405247814}
  }
```


category extraction result는 Aspect Category Detection (ACD)에 대해서만 평가한것이고, entire pipeline은 ASC까지 포함한 성능입니다.
| 평가                       |  P/R/F1         |
| ---------------------------- | -------------- |
| category extraction result | 0.62/0.46/0.53 |
| entire pipeline result | 0.58/0.43/0.49 |


## reference
xlm-roberta-base in huggingface (https://huggingface.co/xlm-roberta-base)

모두의말뭉치 in 국립국어원 (https://corpus.korean.go.kr/)
## Authors
- 정용빈, Teddysum, ybjeong@teddysum.ai
