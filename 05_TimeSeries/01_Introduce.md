시계열 데이터란 무엇인가?
일정 시간 간격으로 배치된 데이터들의 수열입니다.
어떤 순서를 내포하고 있습니다.
시간적으로 종속된 모든 데이터는 시계열에 해당된다고 볼 수 있습니다.
시계열 분석이란?
시간 순서대로 정렬된 데이터에서 의미있는 요약과 통계정보를 추출하기 위한 노력입니다.
이는 예측에 주로 활용되나, 과거의 행동을 진단하는 과정을 포함하고 있습니다.
즉, "과거가 미래에 어떤 영향을 주는가?", "과거와 미래에 어떻게 연관되어 있는가?" 의 해답을 찾아가는 과정!

1-3. 시계열 데이터의 이해




0:00
/
7:08

1
x



용어 설명
용어	설명
시계열 데이터(Time Series data)	일정한 시간 간격으로 규칙성이 있는 것처럼 보여지는 데이터이며 주로 과거와 현재를 통해 앞으로 일어날 미래를 예측하는 용도로 사용됩니다. 심전도, 월별 또는 계절별 온도 및 수온 데이터가 대표적입니다.
심전도(Electrocardiogram)	심장 박동 시 심장을 통과하는 전류가 기록된 데이터입니다. ECG 및 EKG로 부릅니다.
예측(Prediction)	머신러닝(Machine Learning)에서 의미하는 예측은 입력값(input 혹은 x)을 모델 및 알고리즘(함수)에 넣어서 나온 것들을 의미합니다. 출력(output)값 혹은 예측된 정답 (
y
^
y
^
​
  = y hat) 이라고도 하며 주로 predict, pred으로도 혼용합니다.
곰페르츠 모형(Gompertz)	유통업에서의 곰페르츠 모형은 어떤 특정한 제품이 시장에서 얼마나 팔릴 수 있는지 추정하는 모델입니다.
시계열 데이터 둘러보기
① 의료 분야의 시계열

Natural and Political Observations Made upon the Bills of Mortality
EKG(ECG) 데이터
② 금융 분야의 시계열

주가 예측 문제
③ 리테일 분야의 시계열

유통업의 수요 예측
④ 시계열 데이터 예시

다양한 시계열 데이터 예시
의료 분야의 시계열
‘Natural and Political Observations Made upon the Bills of Mortality’
1676년도에 초판이 발행된 최초의 의료 시계열 데이터를 다룬 기록이 있습니다.
누군가 사망하면 교회의 종이 울리는 것을 바탕으로 그들의 죽음을 사후적으로 추적하여 사망한 사람의 수를 시계열적으로 기록하였습니다.
EKG(ECG) 데이터
전기신호를 통해 심장의 상태를 진단하는 기술로 심장 상태에 대한 정보를 시계열적으로 제공합니다.
금융 분야의 시계열
주가 예측 문제
단기(Short Term) vs 장기(Long Term)
시장 지수 예측 vs 개별 주가 예측
리테일 분야의 시계열
유통업 물량 수요 예측
시계열 분석과 다양한 정성적 기법을 통합하여 미래의 수요를 예측하는 방법은 실무적으로 항상 이용되고 있습니다.
대표적으로 곰페르츠 모형 등이 존재합니다.
그 외 다양한 시계열 예시
시계열 데이터는 기본적으로 선 그래프(Line Plot)로 나타내며 아래와 같이 나타납니다.

Electrical activity in the brain
뇌전도 검사(EEG, Electroencephalogram) : 뇌파 또는 뇌의 전기적 활동의 이상을 감지하는 검사입니다.
content img
[EEG]
https://www.researchgate.net/figure/Example-of-microsleep-periods-EEG-channel-A2-C3-C4-Cz-O1-O2-Fp1-and-Fp2-ECG_fig1_4268345
Rainfall measurements
강수량 예측 : 측정장비인 우량계를 사용하여 강우심도와 강우강도를 측정하며 측정된 수치를 기록하여 데이터로 만들고 초단기 강수량, 월별, 연도별, 계절별 강수량을 분석할 수 있습니다.
content img
[한국 52개 관측소 일강수량]
https://www.ibs.re.kr/cop/bbs/BBSMSTR_000000000738/selectBoardArticle.do?nttId=19050
Stock prices
주가 예측 : 장단기 주가를 예측합니다. 대부분의 주가 예측 데이터를 확인해보면 Box plot과 비슷한 candle chart를 사용합니다. candle chart는 거래자가 지정한 시가, 종가, 고가, 저가를 표시하고 있으며 과거 패턴을 기반으로 가격의 움직임을 결정하는데 사용합니다.
content img
[네이버 KOSPI 월봉]
https://m.stock.naver.com/domestic/index/KOSPI/total
Number of sunspots
태양의 흑점 수 : 지구에 영향을 주는 태양풍과 우주의 물리적 현상을 이해하기 위해 태양 표면에 있는 흑점을 측정하여 활용합니다.
content img
[태양 표면에 존재하는 흑점 및 흑점 그룹수를 나타낸 시계열 데이터]
https://en.wikipedia.org/wiki/Wolf_number#/media/File:Zurich_sunspot_number_since_1750.png
Annual retail sales
연간 소매 판매 : 소매 단계에서 총 판매액의 변동을 측정하여 기업에서 매출 상승을 위한 판매 전략이나 상권 변화를 예측하는 등의 분석 지표로서 활용할 수 있습니다.
content img
[Korea annual retail sales]
https://www.theglobaleconomy.com/South-Korea/retail_sales_y_on_y/
Monthly subscribers
월 구독자수 : 실시간으로 일, 월 평균 구독자 증가 및 하락을 통해 채널의 성장을 예측할 수 있습니다. 또한 영상의 조회수 분석을 통해 어떤 영상이 인기 있는지를 판단하고 채널이 성장할 수 있도록 지표로 사용할 수 있습니다.
content img
[유튜브 스튜디오 Lifetime 구독자 수]
https://www.peggyktc.com/2019/08/view-youtube-subscriber-count.html
Heartbeats per minute
분당 심장 박동수 : 심장이 올바르게 뛰고 있는지를 통해 심장 질환을 미리 예측할 수 있습니다.
content img
[ECG Rate Interpretation]
https://litfl.com/ecg-rate-interpretation/