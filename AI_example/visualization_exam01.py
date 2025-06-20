# -*- coding: utf-8 -*-
"""visualization_exam01.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MuugsKceFxMH58OuWRLzpTlNFmzILLEY

시각화(plot)!!!!
"""

import numpy as np
import matplotlib # 시각화 플롯을 묶어 놓은 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns # matplotlib보다 좀 예쁘고 디테일한 기능을 제공하는 라이브러리

# Line plot

point = 100
x = list(range(100)) # 0~99
print(x)
y = list(range(100))

fig, axes = plt.subplots(figsize=(7, 4)) # 그림 크기 설정
axes.plot(x, y, color='red', linewidth=2) # line의 색과 두께 설정
plt.show()

point = 100
x = np.asarray(list(range(point))) # 리스트를 행렬처럼 변환, numpy의 array 타입은 사칙연산이 가능하다.
print(x)
y = 2 * x * x + 3
fig, axes = plt.subplots(figsize=(7, 4)) # 그림 크기 설정
axes.plot(x, y, color='blue', linewidth=2) # line의 색과 두께 설정
plt.show()

point = 100
x = np.asarray(list(range(-50, 50))) # 리스트를 행렬처럼 변환, numpy의 array 타입은 사칙연산이 가능하다.
print(x)
y = 2 * x * x + 3
fig, axes = plt.subplots(figsize=(7, 4)) # 그림 크기 설정
axes.plot(x, y, color='blue', linewidth=2) # line의 색과 두께 설정
plt.axhline(color='black', linewidth=1) # horizental line
plt.axvline(color='black', linewidth=1) # vertical line
plt.show()

"""---

4/14 수업
"""

labels = 'Frogs', 'Hogs', 'Dogs' # 각 조각의 라벨 지정
sizes = [15, 15, 45] # 각 조각의 크기(비율) 설정
explode = (0, 0, 0) # 각 파이 조각을 중심으로부터 얼마나 띄워낼 것인가?
fig, axes = plt.subplots(figsize = (7, 4)) # axes: 실제 그래프를 그리는 영역
# %를 출력하고 싶으면 %%로 해야한다., shadow = True -> 약간 입체적으로 보임.
# wedges: 파이 조각 객체 리스트, texts: 조각 라벨, autotexts: 퍼센트 출력 텍스트
wedges, texts, autotexts = axes.pie(sizes, labels=labels, autopct='%1.2f%%',
                                    shadow=True, startangle=90, explode = explode, counterclock=False)
               # autopct: 퍼센트 출력, shadow: 그림자 추가, startangle: 시계방향 회전 기준으로 시작 각도 설정
               # counterclock: False면 시계방향으로 회전

# 조각의 스타일 커스터마이징
wedges[1].set(hatch = '///') # 두 번째 조각에 해칭
wedges[0].set_radius(1.1) # 첫 번째 조각의 반지름을 1.1배로 -> 조금 더 튀어나온 모양
axes.axis('equal') # x축과 y축을 동일하게 설정(안 찌그러지도록)
plt.show()

size = 0.3
vals1 = [40, 35, 25]
vals2 = [11, 29, 12, 23, 12 ,13]
cmap = plt.get_cmap('tab20c')
outer_colors = cmap([0, 4, 12])
inner_colors = cmap([1, 2, 5, 6, 13, 14])

fig, ax = plt.subplots()
ax.pie(vals1, radius =1, colors = outer_colors,
       labels=['man', 'woman', 'child'], autopct='%1.1f%%', pctdistance=0.85,
       wedgeprops = dict(width = size, edgecolor = 'w'))

ax.pie(vals2, radius =1-size, colors = outer_colors,
       autopct='%1.1f%%', pctdistance=0.8,
       wedgeprops = dict(width = size, edgecolor = 'w'))
plt.show()

from matplotlib import colormaps
list(colormaps)

"""https://matplotlib.org/stable/users/explain/colors/colormaps.html"""

import matplotlib as mpl # matplotlib의 저수준 api를 사용하기 위해 import, 주로 색상 맵 쓸 때 사용

# matplot color map 확인
cmaps = {} # 색상 맵을 분류해서 담을 딕셔너리
gradient = np.linspace(0, 1, 256) # 1차원 배열의 구간은 256가지
gradient = np.vstack((gradient, gradient)) # 1차원 배열을 위아래로 2번 쌓아서 2줄짜리의 2차원 배열 생성
                                          # -> 기다란 네모 모양의 색상 맵을 표시할 수 있음

def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list

plot_color_gradients('Perceptually Uniform Sequential',
                     ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

plot_color_gradients('Sequential',
                     ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])

for name, hex_value in mpl.colors.cnames.items():
  print(name, hex_value)

# 플롯 스타일 테마 설정
plt.style.use('seaborn-v0_8') # seaborn과 유사한 테마로 설정

plt.style.available

"""막대그래프 그리기!!!!!"""

n = 10
x = list(range(n)) # x축 좌표를 0부터 9까지로 설정
y1 = np.random.uniform(1, 7, n) # 1~6까지의 값을 균등분포 기준으로 n개 생성
y2 = np.random.uniform(1, 7, n)
fig, axes = plt.subplots(figsize = (5, 3))
axes.bar(x, y1, facecolor = 'olive')
axes.bar(x, -y2, facecolor = 'darkblue')
axes.set_title('Bar plot', fontdict = {'fontsize' : 15})

plt.show()

width = 0.35 # 막대의 너비는 0.35
fig, axes = plt.subplots(figsize = (5, 3))
axes.bar(list(map(lambda x:x - width/2,x )), y1, width, label = 'y1')
axes.bar(list(map(lambda x:x + width/2,x )), y2, width, label = 'y2')

plt.show()
# 해석해보기

"""타이타닉 데이터셋으로 실습"""

import pandas as pd

raw_data = sns.load_dataset('titanic')
raw_data.head()

titanic_age = raw_data[['age', 'survived']]
titanic_age = titanic_age.dropna()
titanic_age.head()

def age_band(num):
    for i in range(1, 10):
        if num < 10 * i:
            return f'under {i}0s'
    return '90s'

titanic_age['age_band'] = titanic_age['age'].apply(age_band)
titanic_age.head()

titanic_age = titanic_age[['age_band', 'survived']]
print(titanic_age.head())

titanic_group_age = titanic_age.groupby('age_band')['survived'].mean()
print(titanic_group_age)

# subplots(1,2) -> 1행 2열
fig, axes = plt.subplots(1, 2, figsize = (12, 5))
axes[0].bar(titanic_group_age.index, titanic_group_age)
axes[0].set_title('Age Band & Servival Rate')

color_map = ['gray'] * 9
color_map[0] = color_map[8] = 'red'

axes[1].bar(titanic_group_age.index, titanic_group_age,
             color = color_map, edgecolor = 'black',
            linewidth = 1.2, alpha = 0.7)
# font 크기 조절, fontweight에 bold(진하게) 적용, 위치포지션은 없어도 거의 동일..?
axes[1].set_title('Age Band & Servival Rate', fontsize = 15, fontweight = 'bold', position = (0.5, 1.1))
for i, rate in enumerate(titanic_group_age):
    axes[1].annotate(f'{rate*100:.2f}%', xy = (i, rate), ha = 'center', va='bottom',
                    fontweight = 'bold', color = '#383838') # annotate는 plot에 원하는 문자열을 출력할 때 사용한다.
axes[1].set_xticklabels(titanic_group_age.index, rotation=45) # x축 라벨을 45도 돌리기
plt.show()

# subplots(1,2) -> 1행 2열
fig, axes = plt.subplots(1, 2, figsize = (12, 5))
axes[0].bar(titanic_group_age.index, titanic_group_age)
axes[0].set_title('Age Band & Servival Rate')

color_map = ['gray'] * 9
color_map[0] = color_map[8] = 'red'
with plt.xkcd(2):
  axes[1].bar(titanic_group_age.index, titanic_group_age,
              color = color_map, edgecolor = 'black',
              linewidth = 1.2, alpha = 0.7) # alpha는 막대그래프의 투명도를 의미
  # font크기 키우고, fontweight에 bold(진하게) 적용, 위치포지션은 없어도 거의 동일..?
  axes[1].set_title('Age Band & Servival Rate', fontsize = 15, fontweight = 'bold', position = (0.5, 1.1))
  for i, rate in enumerate(titanic_group_age):
      axes[1].annotate(f'{rate*100:.2f}%', xy = (i, rate), ha = 'center', va='bottom',
                      fontweight = 'bold', color = '#383838') # annotate는 plot에 원하는 문자열을 출력할 때 사용한다.
  axes[1].set_xticklabels(titanic_group_age.index, rotation=45) # x축 라벨을 45도 돌리기
plt.show()

"""4/15

산점도!!!!!!!
"""

# 초기 세팅
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
iris_df.info()
iris_df.head()

fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1행, 2열
axes[0].scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'], # 0번(1번째) 열에 산점도를 그림
                c='gray')
axes[0].set_title('iris')
axes[1].scatter(x='petal length (cm)', y='petal width (cm)', label='setosa',
                data=iris_df[iris_df['species']=='setosa'], c='olive', s=20, alpha=0.5)
axes[1].scatter(x='petal length (cm)', y='petal width (cm)', label='versicolor',
                data=iris_df[iris_df['species']=='versicolor'], c='plum', s=20, alpha=0.5)
axes[1].scatter(x='petal length (cm)', y='petal width (cm)', label='virginica',
                data=iris_df[iris_df['species']=='virginica'], c='navy', s=20, alpha=0.5)
axes[1].legend()
axes[1].set_title('iris')
plt.show()

# seaborn으로 산점도 그리기
sns.FacetGrid(iris_df, hue='species', height=4).map(plt.scatter,
              'petal length (cm)', 'petal width (cm)').add_legend()
plt.show()

# pandas로 산점도 그리기
pd.plotting.scatter_matrix(iris_df, c=iris.target,
                           figsize=(8,8), s=20, marker='0', alpha=0.3)

sns.pairplot(iris_df, hue='species')
fig = plt.gcf() # get current figure
fig.set_size_inches(8, 8) # 사이즈 줄이기
plt.show()

# 산점도를 3차원으로
from plotly.express import scatter_3d

fig = scatter_3d(iris_df, x='sepal length (cm)', y='petal length (cm)', z='petal width (cm)',
                 color='sepal width (cm)', symbol='species')
fig.show()

"""box plot!!!"""

plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1) # 2행 2열 1번
sns.boxplot(x='species', y='petal length (cm)', data=iris_df)
plt.subplot(2, 2, 4) # 2행 2열 4번
sns.boxplot(x='species', y='petal width (cm)', data=iris_df)
plt.show()

plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1)
sns.boxplot(x='species', y='petal length (cm)', data=iris_df, showmeans=True)
sns.swarmplot(x='species', y='petal length (cm)', data=iris_df, color='k', alpha=0.3)
plt.subplot(2, 2, 2)
sns.boxplot(x='species', y='petal width (cm)', data=iris_df, showmeans=True)
sns.swarmplot(x='species', y='petal length (cm)', data=iris_df, color='k', alpha=0.3)
plt.subplot(2, 2, 3)
sns.boxplot(x='species', y='sepal length (cm)', data=iris_df, showmeans=True)
sns.swarmplot(x='species', y='petal length (cm)', data=iris_df, color='k', alpha=0.3)
plt.subplot(2, 2, 4)
sns.boxplot(x='species', y='sepal width (cm)', data=iris_df, showmeans=True)
sns.swarmplot(x='species', y='petal length (cm)', data=iris_df, color='k', alpha=0.3)
plt.show()

"""박스 플롯과 비슷하게 생긴 바이올린 플롯!!"""

plt.figure(figsize=(10, 7))
plt.subplot(221) # 221 == (2, 2, 1)
sns.violinplot(x='species', y='petal length (cm)', data=iris_df)
plt.show()

plt.figure(figsize=(10, 7))
plt.subplot(221)
sns.violinplot(x='species', y='petal length (cm)', data=iris_df)
plt.subplot(222)
sns.violinplot(x='species', y='petal width (cm)', data=iris_df)
plt.subplot(223)
sns.violinplot(x='species', y='sepal length (cm)', data=iris_df)
plt.subplot(224)
sns.violinplot(x='species', y='sepal width (cm)', data=iris_df)
plt.show()

"""히스토그램!!!"""

n_point = 100
n_bins = 10

dist1 = np.random.normal(0, 2, n_point) # 정규분포

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
axes[0].hist(dist1, bins=n_bins)
plt.show()

n_point = 1000000
n_bins = 1000

dist1 = np.random.normal(0, 4, n_point) # 정규분포, 표준편차 4, 정규분포를 따르는 값을 랜덤으로 뽑기

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
axes[0].hist(dist1, bins=n_bins) # 히스토그램으로 그리기
plt.show()

n_point = 100000000
n_bins = 50

dist1 = np.random.normal(0, 2, n_point)
dist2 = np.random.normal(0, 10, n_point)

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
axes[0].hist(dist1, bins=n_bins)
axes[1].hist(dist2, bins=n_bins)
plt.show()

"""히트맵!!!!!!!!"""

fig, axes = plt.subplots(1, 2, figsize=(7, 4))
axes[0].hist2d(dist1, dist2, bins=n_bins, density=True)
axes[0].axis('equal')
axes[1].hist2d(dist1, dist2, bins=n_bins, cmap='Blues')
plt.show()

fig = plt.figure(figsize=(16, 10))
axes = fig.add_subplot(projection='3d') # 3d plot 축 생성
# 각 셀(bin)에 해당하는 데이터 개수, x축의 구간 경계 값, y축의 구간 경계 값 반환
hist, xedges, yedges = np.histogram2d(dist1, dist2, bins=n_bins,
                                      range=[[-7, 7], [-30, 30]]) # 범위: x축은 [-7, 7], y축은 [-30, 30]
# x, y 위치 좌표 계산
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing='ij')
xpos = xpos.ravel() # 2차원 배열을 1차원 배열로 펼치기
ypos = ypos.ravel()
zpos = 0 # z축 시작 높이
dx = dy = 0.3 * np.ones_like(zpos) # 막대의 너비를 0.3으로 고정
dz = hist.ravel()

axes.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average') # 막대의 시작 위치, 크기, 정렬 기준을 설정
axes.view_init(elev=20, azim=15) # 뷰 포인트 설정
plt.show()

"""여러 플롯을 나열하기(subplots() 사용하기)"""

x = [0, 1]
y = [0, 1]

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes[0, 1].scatter(x='petal length (cm)', y='petal width (cm)', label='setosa',
                data=iris_df[iris_df['species']=='setosa'], c='olive', s=20, alpha=0.5)
axes[0, 1].scatter(x='petal length (cm)', y='petal width (cm)', label='versicolor',
                data=iris_df[iris_df['species']=='versicolor'], c='plum', s=20, alpha=0.5)
axes[0, 1].scatter(x='petal length (cm)', y='petal width (cm)', label='virginica',
                data=iris_df[iris_df['species']=='virginica'], c='navy', s=20, alpha=0.5)
axes[1, 1].plot(x, y)
plt.show()

"""figure()"""

fig = plt.figure(figsize=(8, 5))
axes = fig.add_axes([0, 0, 1, 1]) # 플롯이 피규어에 배치될 위치 지정
axes.plot(x, y)
plt.show()

fig = plt.figure(figsize=(8, 5))
axes = [None] * 3 # None 3개가 들어 있는 리스트 생성
axes[0] = fig.add_axes([0.1, 0.1, 0.4, 0.8])
axes[1] = fig.add_axes([0.55, 0.15, 0.35, 0.4])
axes[2] = fig.add_axes([0.65, 0.66, 0.2, 0.3])
for i in range(3):
  axes[i].set_title('axes[{}]'.format(i)) # for문을 이용해서 각 axes에 타이틀 붙이기

plt.show()

# plot 위치 지정

fig = plt.figure(figsize=(8, 5), tight_layout=True)
axes = [None] * 3 # None 3개가 들어 있는 리스트 생성
axes[0] = fig.add_axes([0.1, 0.1, 0.4, 0.8])
axes[1] = fig.add_axes([0.55, 0.15, 0.35, 0.4])
axes[2] = fig.add_axes([0.65, 0.66, 0.2, 0.3])
axes[2].plot(x, y)
for i in range(3):
  axes[i].set_title('axes[{}]'.format(i)) # for문을 이용해서 각 axes에 타이틀 붙이기
  axes[i].set_xticks([]) # x축 눈금 없애기
  axes[i].set_yticks([]) # y축 눈금 없애기
plt.show()

# 그리드

fig = plt.figure(figsize=(8, 5))
axes = []
axes.append(plt.subplot2grid((2, 3), (0, 0)))
#axes.append(plt.subplot2grid((2, 3), (0, 1)))
axes.append(plt.subplot2grid((2, 3), (0, 2)))
#axes.append(plt.subplot2grid((2, 3), (1, 0)))
axes.append(plt.subplot2grid((2, 3), (1, 1)))
#axes.append(plt.subplot2grid((2, 3), (1, 2)))
for i in range(len(axes)):
  axes[i].set_title('axes[{}]'.format(i))
  axes[i].set_xticks([])
  axes[i].set_yticks([])
plt.show()

fig = plt.figure(figsize=(8, 5))
axes = []
axes.append(plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2))
axes.append(plt.subplot2grid((3, 4), (0, 2)))
axes.append(plt.subplot2grid((3, 4), (0, 3), rowspan=3))
axes.append(plt.subplot2grid((3, 4), (1, 2)))
axes.append(plt.subplot2grid((3, 4), (2, 0), colspan=3))
for i in range(len(axes)):
  axes[i].set_title('axes[{}]'.format(i))
  axes[i].set_xticks([])
  axes[i].set_yticks([])
plt.show()

# subplot의 그리드 배치를 슬라이싱으로 간단?하게 표현하기

fig = plt.figure(figsize=(8, 5), tight_layout=True)
axes = []
gs = fig.add_gridspec(3, 4)

axes.append(fig.add_subplot(gs[0:2, 0:2]))
axes.append(fig.add_subplot(gs[0, 2]))
axes.append(fig.add_subplot(gs[:, 3]))
axes.append(fig.add_subplot(gs[1, 2]))
axes.append(fig.add_subplot(gs[2, :-1]))
for i in range(len(axes)):
  axes[i].set_title('axes[{}]'.format(i))
  axes[i].set_xticks([])
  axes[i].set_yticks([])
plt.show()