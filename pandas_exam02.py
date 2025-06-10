import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 15)
# pd.set_option('display.max_rows', 400)
df=pd.read_csv('./datasets/auto-mpg.csv',
               names=['mpg', 'cylinders', 'displacement', 'horsepower',
                      'weight', 'acceleration', 'model yoear', 'origin', 'name'])

print(df.head(10))
print(df.tail(10))
print(df.shape)
df.info()
print(df.dtypes)
print(dir(df))
print(df.describe(include='all').T)
print(df.mean(numeric_only=True)) # 평균 보기
print(df.max()) # 최댓값 보기
print(df.std(numeric_only=True)) # 표준편차 보기
print(df.corr(numeric_only=True)) # 상관계수 보기(1(절댓값)에 가까울 수록 상관계수가 높다. 0에 가까울 수록 상관이 그다지 없다... 1이냐 -1이냐는 방향 차이)


# mpg는 감이 안 오니까 kpl로 바꿔 보자(단위 변환)
mpg_to_kpl = 0.425144
df['kpl'] = df['mpg'] * mpg_to_kpl # 'kpl'이라는 column 새로 생성
df['kpl'] = df['kpl'].round(2)
print(df.head(30))

# 결측치 데이터를 어떻게든 해 보자
print(df.horsepower.unique())
print(df.horsepower.value_counts().head(30))

df.horsepower = df.horsepower.replace('?', np.nan)
df.dropna(subset=['horsepower'], axis=0, inplace=True) # 'horsepower'에 있는 NaN 값을 드롭한다. axis=0(row)에 대해서
df.horsepower = df.horsepower.astype('float') # type 바꾸기
df.info()

# origin을 정수가 아닌 이름으로 바꾸기
print(df['origin'].unique()) # 고유값끼리 보기
print(df.origin.value_counts()) # 각 고유값이 몇 개 있는지

df.origin = df.origin.astype('category') # category 타입으로 바꾸기
df.info()
print(df.origin)

df.origin = df.origin.replace({1:'US', 2:'EU', 3:'JP'})
print(df.origin)

df.origin = df.origin.astype('string') # category가 어려우면 string으로 해도 돼...
df.info()

# horsepower를 category화하기
count, bin_dividers = np.histogram(df['horsepower'], bins=3) # 최솟값, 최댓값을 찾아서 3개의 구간으로 나눈다.
print(count)
print(bin_dividers)

bin_names = ['저출력', '보통출력', '고출력']
df['hp_bin'] = pd.cut(x=df['horsepower'], bins=bin_dividers,
                      labels=bin_names, include_lowest=True)
print(df[['horsepower','hp_bin']])
df.info()

# 가변수화(더미화)
# One-Hot Encoding
df1 = pd.get_dummies(df.hp_bin) # One-Hot 인코딩 형태로 변환
print(df1.head(30))

# 중복 제거
# 예시로 중복 데이터가 있는 데이터프레임을 생성해 보았다.
df = pd.DataFrame({'c1':['a', 'a', 'b', 'a', 'b'],
                   'c2':[1, 1, 1, 2, 2],
                   'c3':[1, 1, 2, 2, 2]})
print(df)

df_dup = df.duplicated() # 중복 여부를 반환한다
print(df_dup)

df2 = df.drop_duplicates() # 중복을 제거한 데이터프레임을 반환한다
print(df2)

# 'c2'를 기준으로 중복 제거하기
df_dup = df['c2'].duplicated()
print(df_dup)

df2 = df.drop_duplicates(subset='c2')
print(df2)

# 인덱스 리셋
# df2.reset_index(inplace=True) # 기존 인덱스 정보 저장하기
# print(df2)
df2.reset_index(drop=True, inplace=True) # 기존 인덱스 정보 drop
print(df2)