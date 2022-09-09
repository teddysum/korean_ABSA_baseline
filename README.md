# korean_ABSA_baseline

sample.jsonl은 국립국어원에서 제공한 데이터의 일부분이며, 전체 데이터는 국립국어원 모두의 말뭉치에서 다운받으실 수 있습니다. https://corpus.korean.go.kr/

학습된 baseline 모델은 아래 링크에서 받으실 수 있습니다.

category extraction model link: https://drive.google.com/file/d/109z2rAk53UTVzFjRu-KtN_E3TiZ4yx-r/view?usp=sharing

polarity classification model link: https://drive.google.com/file/d/1UU1I9CR4TIEYRD87PAhvOJ763AjE-ZDT/view?usp=sharing

## 모델 구성
category extraction model과 polarity classification 모델을 pipeline으로 연결한 모델입니다.

xlm-roberta-base를 기반으로 학습하였습니다.

### category extraction
모델 입력형태를 \<s>sentence_form\</s>\</s>카테고리\</s>와 같이하고, 각 category별로 해당 문장에서 추출 될지/ 되지 않을지에 대한 이진 분류를 합니다.
모든 카테고리에 대해 수행한 뒤, True로 판별된 category를 모으면, 해당 문장에서 나타난 카테고리가 모아집니다.

### polarity classification
추출 된 category에 대해 모델 입력형태를 \<s>sentence_form\</s>\</s>카테고리\</s>와 같이하고, positive, neutral, negative로 classification 합니다.

### 성능
baseline 코드에서 제공된 평가 코드로 평가하였을때, 아래와 같이 결과가 나왔습니다.
category extraction result는 category 추출에 대해서만 평가한것이고, entire pipeline은 polarity 분류까지 포함한 성능입니다.
| 평가                       |  P/R/F1         |
| ---------------------------- | -------------- |
| category extraction result | 0.42/0.40/0.41 |
| entire pipeline result | 0.39/0.38/0.38 |
