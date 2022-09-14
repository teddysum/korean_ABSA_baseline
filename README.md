# korean_ABSA_baseline

본 소스코드는는 '2022 국립국어원 인공 지능 언어 능력 평가'의 속성 기반 감성 분석 과제의 베이스라인 모델 및 학습과 평가를 위한 코드를 제공하고 있습니다. 자세한 코드의 설명은 '...ipynb' notebook을 확인해주세요



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


## 모델 구성
Aspect Category Detection (ACD) 모델과 Aspect Sentiment Classification (ASC) 모델을 pipeline으로 연결한 모델입니다.

xlm-roberta-base를 기반으로 학습하였습니다.

학습된 baseline 모델은 아래 링크에서 받으실 수 있습니다.

Aspect Category Detection (ACD) model link: https://drive.google.com/file/d/109z2rAk53UTVzFjRu-KtN_E3TiZ4yx-r/view?usp=sharing

Aspect Sentiment Classification (ASC) link: https://drive.google.com/file/d/1UU1I9CR4TIEYRD87PAhvOJ763AjE-ZDT/view?usp=sharing

### category extraction
모델 입력형태를 \<s>sentence_form\</s>\</s>카테고리\</s>와 같이하고, 각 category별로 해당 문장에서 추출 될지/ 되지 않을지에 대한 이진 분류를 합니다.
모든 카테고리에 대해 수행한 뒤, True로 판별된 category를 모으면, 해당 문장에서 나타난 카테고리가 모아집니다.

### polarity classification
추출 된 category에 대해 모델 입력형태를 \<s>sentence_form\</s>\</s>카테고리\</s>와 같이하고, positive, neutral, negative로 classification 합니다.

### 성능
baseline 코드에서 제공된 평가 코드로 평가하였을때, 아래와 같이 결과가 나왔습니다.
ACD result는 category 추출에 대해서만 평가한것이고, entire pipeline은 ASC까지 포함한 성능입니다.
| 평가                       |  P/R/F1         |
| ---------------------------- | -------------- |
| category extraction result | 0.42/0.40/0.41 |
| entire pipeline result | 0.39/0.38/0.38 |


## Authors
- 정용빈, Teddysum, ybjeong@teddysum.ai
