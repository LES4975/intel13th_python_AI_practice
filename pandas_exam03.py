import pandas as pd

df  = pd.DataFrame({'ten':[10, 20, 30, 40, 50],
                    'one':[1, -2, -4, 5, 6]})
print(df)

# 데이터프레임에 함수 적용하는 방법 1
# 여기서 적용할 함수는 ReLU다.
# for i in range(len(df)):
#     if df.iloc[i, 1] < 0 :
#         df.iloc[i, 1] = 0

print(df)

def relu(value):
    if value < 0 : value = 0
    return value

df.one = df.one.apply(relu)
print(df)

# apply를 좀 더 사용해 보자! lambda 함수를 써서...
sr1 = df.ten.apply(lambda x: x * 3 - 4)
print(sr1)

# 이어 붙이기
df1 = pd.DataFrame({'a':['a0', 'a1', 'a2', 'a3'],
                    'b':['b0', 'b1', 'b2', 'b3'],
                    'c':['c0', 'c1', 'c2', 'c3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'a':['a2', 'a3', 'a4', 'a5'],
                    'b':['b2', 'b3', 'b4', 'b5'],
                    'c':['c2', 'c3', 'c4', 'c5'],
                    'd':['d2', 'd3', 'd4', 'd5']},
                   index=[2, 3, 4, 5])

print(df1)
print(df2)

result = pd.concat([df1, df2])
# result.reset_index(drop=True, inplace=True)
print(result)

#concat 함수에서 인덱스 리셋하기
result1 = pd.concat([df1, df2], ignore_index=True)
print(result1)

# column 방향으로 데이터프레임 이어붙이기
result2 = pd.concat([df1, df2], axis='columns')
print(result2)

# inner join
result3 = pd.concat([df1, df2], axis='columns', join='inner')
print(result3)

# 이번엔 또 무슨 짓을...
# 타이타닉 데이터셋으로 머신러닝

import seaborn as sns

raw_data = sns.load_dataset('titanic')
print(raw_data.head())
raw_data.info()

# 결측치 제거합쉬다

print(raw_data.isnull().sum()) # 전체 결측치 갯수 확인하기

# column 제거하기
clean_data = raw_data.dropna(axis=1, thresh=500) # NULL 값이 500개 이상인 column 지우기
clean_data.info()

# 'age' 좀 어떻게 하기
mean_age = clean_data['age'].mean()
print(mean_age)
print(clean_data.head(10))
clean_data['age'].fillna(mean_age, inplace=True) # NULL 값을 평균값으로 채우기
print(clean_data.head(10))

# 'embarked'나 'embark_town' 좀 어떻게 하기 -> 이전 값으로 채우기
print(clean_data.embarked[825:830])
clean_data['embarked'].fillna(method='ffill', inplace=True) # NULL 값은 바로 이전(앞)의 데이터 그대로 쓰기
print(clean_data.embarked[825:830])

print(clean_data.isnull().sum()) # NULL 값 다 제거되었는지 확인해 봄

# 'who' 별로 생존률 살펴 보기
print(clean_data['who'].unique())
print(clean_data['who'].value_counts())

grouped = clean_data.groupby('who')
print(grouped)

grouped_df = {}
for key, group in grouped:
    print(key)
    print(len(group))
    grouped_df[key] = group
    print(group.who.head())

print(grouped_df.keys())
print(grouped_df['man'].survived.mean())
print(grouped_df['woman'].survived.mean())
print(grouped_df['child'].survived.mean())

# 'pclass' 별로 생존률 살펴 보기
print(clean_data['pclass'].unique())
print(clean_data['pclass'].value_counts())

grouped = clean_data.groupby('pclass')
print(grouped)

grouped_df = {}
for key, group in grouped:
    print(key)
    print(len(group))
    grouped_df[key] = group
    print(group.who.head())

print(grouped_df.keys())
average = grouped.mean(numeric_only=True)
print(average)

#최솟값
min = grouped.min(numeric_only=True)
print(min)
# 최댓값
max = grouped.max(numeric_only=True)
print(max)

grouped_two = clean_data.groupby(['pclass', 'sex']) # column을 두 가지 주면, 각 column의 unique값에 해당하는 조합의 그룹들이 만들어진다
for key, group in grouped_two:
    print(key)
    print(len(group))
    print(group.head())

group3f = grouped_two.get_group((3, 'female')) # 3등실의 여성 그룹만 뽑자
print(group3f)
print(type(group3f))

print(grouped_two.mean(numeric_only=True))

# 내가 보고 싶은 데이터만 확인하기
grouped_two = clean_data.groupby(['pclass', 'who'])
for key, group in grouped_two:
    print(key)
    print(len(group))
    print(group.head())

print(grouped_two.mean(numeric_only=True).loc[(3, 'woman')])

# 피벗 테이블
pdf1 = pd.pivot_table(clean_data,
                      index='pclass',
                      columns='sex',
                      values='age',
                      aggfunc='mean')
print(pdf1)

pdf2 = pd.pivot_table(clean_data,
                      index='pclass',
                      columns='sex',
                      values=['age', 'fare'],
                      aggfunc='mean')
print(pdf2)

pdf3 = pd.pivot_table(clean_data,
                      index='pclass',
                      columns='sex',
                      values='age',
                      aggfunc=['mean', 'sum', 'std'])
print(pdf3)

pdf4 = pd.pivot_table(clean_data,
                      index=['pclass', 'sex'], # 이중 인덱스
                      columns='survived',
                      values=['age', 'fare'],
                      aggfunc=['mean', 'max'])
print(pdf4)
print(pdf4.index)
print(pdf4.columns)

# 조건 인덱싱
# 데이터프레임에서 10대만을 인덱싱해 보기
df_teenage = clean_data.loc[(clean_data['age'] >= 10) & (clean_data['age'] <=19)] # 조건을 걸어서 자료들을 추려 보자
# 오잉? 그런데 비트 연산자를 쓰네?
print(df_teenage)

print(len(df_teenage))
print(df_teenage.survived.sum())
print(df_teenage.survived.mean())

# 이번에는 노인만을 인덱싱해 보자.
old_man = clean_data.loc[(clean_data['age'] >= 60)]
print(old_man)
print(len(old_man))
print(old_man.survived.sum())
print(old_man.survived.mean())

# 전체 생존률 볼까?
print(df_teenage.survived.mean())
print(old_man.survived.mean())
print(clean_data.survived.mean())