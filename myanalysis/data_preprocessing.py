import pandas as pd

from config.settings import DATA_DIRS

df = pd.read_csv(DATA_DIRS[0]+'//sizedata.csv');
df.columns=['weight','age','height','size'];
# print(df.info()); # na값 확인
df2 = df.dropna(axis=0, subset=['age','height']); # na값 제거
# df2['height'] = df2['height'].round(1); # 'hegiht'컬럼 반올림.
df2['age'] = df2['age'].astype('int64'); # int형으로 변환
# df2['height'] = df2['height'].astype('int64');
print(df2.info());
# print(df2);
# df2.to_csv("Clothingsizedata.csv", index=False)
# df2.to_csv("Clothingsizedata2.csv", index=False); # 반올림 안한 데이터

