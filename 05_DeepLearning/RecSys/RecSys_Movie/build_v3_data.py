"""
build_v3_data.py
피처 엔지니어링 + Hard Negative Sampling 적용된 v3 데이터 생성 스크립트
"""
import re
import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

print("=" * 50)
print("  v3 데이터 생성 시작")
print("=" * 50)

# ============================================================
# 1. 원본 데이터 로드
# ============================================================
user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
movie_columns = ['movie_id', 'title', 'genres']

users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, names=user_columns, engine='python')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, names=rating_columns, engine='python')
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, names=movie_columns, engine='python', encoding='latin-1')

print(f"Users: {users.shape}, Ratings: {ratings.shape}, Movies: {movies.shape}")

# ============================================================
# 2. 영화 전처리 (기존 v2와 동일)
# ============================================================
movies['movie_year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['movie_decade'] = movies['movie_year'].astype(int).apply(lambda x: str(x - (x % 10)) + 's')
movies['title'] = movies['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x))

genres_split = movies['genres'].str.split('|')
max_genres = genres_split.apply(len).max()
for i in range(max_genres):
    movies[f'genre{i + 1}'] = genres_split.apply(lambda x: x[i] if i < len(x) else None)
movies.drop('genres', axis=1, inplace=True)
movies.fillna('no', inplace=True)

# ============================================================
# 3. 평점 전처리 (기존 v2와 동일)
# ============================================================
ratings['timestamp'] = ratings['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
ratings['rating_year'] = ratings['timestamp'].apply(lambda x: x.split("-")[0])
ratings['rating_month'] = ratings['timestamp'].apply(lambda x: x.split("-")[1])
ratings['rating_decade'] = ratings['rating_year'].astype(int).apply(lambda x: str(x - (x % 10)) + 's')

# ============================================================
# 4. [NEW] 피처 엔지니어링
# ============================================================
print("\n[피처 엔지니어링] 시작...")

# 4-1. 사용자 평균 평점 (구간화)
user_stats = ratings.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
user_stats.columns = ['user_id', 'user_avg_rating', 'user_rating_count']

# 평균 평점을 5구간으로 나누기
user_stats['user_avg_rating'] = pd.cut(
    user_stats['user_avg_rating'], 
    bins=[0, 2, 3, 3.5, 4, 5], 
    labels=['very_low', 'low', 'medium', 'high', 'very_high']
).astype(str)

# 평점 수를 5구간으로 나누기 (quantile 기반)
user_stats['user_rating_count'] = pd.qcut(
    user_stats['user_rating_count'], 
    q=5, 
    labels=['very_few', 'few', 'moderate', 'many', 'very_many']
).astype(str)

print(f"  사용자 통계 피처 생성 완료: {user_stats.shape}")
print(f"  user_avg_rating 분포:\n{user_stats['user_avg_rating'].value_counts()}")
print(f"  user_rating_count 분포:\n{user_stats['user_rating_count'].value_counts()}")

# 4-2. 영화 인기도 (평점 받은 횟수, 구간화)
movie_popularity = ratings.groupby('movie_id')['rating'].count().reset_index()
movie_popularity.columns = ['movie_id', 'movie_popularity']

# 인기도를 5구간으로 나누기 (quantile 기반)
movie_popularity['movie_popularity_bin'] = pd.qcut(
    movie_popularity['movie_popularity'], 
    q=5, 
    labels=['very_unpopular', 'unpopular', 'moderate', 'popular', 'very_popular'],
    duplicates='drop'
).astype(str)

print(f"  영화 인기도 피처 생성 완료: {movie_popularity.shape}")
print(f"  movie_popularity_bin 분포:\n{movie_popularity['movie_popularity_bin'].value_counts()}")

# ============================================================
# 5. [NEW] Hard Negative Sampling
# ============================================================
print("\n[Hard Negative Sampling] 시작...")

# 인기도 기반 가중치 생성 (인기 영화일수록 네거티브로 뽑힐 확률 높음)
movie_pop_dict = dict(zip(movie_popularity['movie_id'], movie_popularity['movie_popularity']))

# label 생성: rating >= 4 → 1, else → 0
ratings['label'] = ratings['rating'].apply(lambda x: x >= 4).astype(int)

# 선호 데이터 (label=1)
positive_ratings = ratings[ratings['label'] == 1].copy()
print(f"  선호 데이터: {positive_ratings.shape[0]}건")

# 사용자가 본 영화 목록
user_seen_movies = ratings.groupby('user_id')['movie_id'].apply(set).to_dict()
unique_movies = set(movies['movie_id'].unique())
all_movie_list = list(unique_movies)

# 각 영화의 인기도 가중치 (정규화)
movie_weights = np.array([movie_pop_dict.get(m, 1) for m in all_movie_list], dtype=float)
movie_weights = movie_weights / movie_weights.sum()

negative_users = []
negative_movies = []
negative_labels = []
negative_rating_years = []
negative_rating_months = []
negative_rating_decades = []

unique_users = users['user_id'].unique()

for user in tqdm(unique_users, desc="Hard Negative Sampling"):
    seen = user_seen_movies.get(user, set())
    if len(seen) < 1:
        continue
    
    # 선호 영화 수
    user_positive_count = len(positive_ratings[positive_ratings['user_id'] == user])
    if user_positive_count < 1:
        continue
    
    # 미시청 영화 추출
    unseen = list(unique_movies - seen)
    if len(unseen) < 1:
        continue
    
    # 샘플 수: 선호 영화 × 5 (기존과 동일 비율)
    sample_size = min(user_positive_count * 5, len(unseen))
    
    # Hard Negative: 인기도 기반 가중치 샘플링
    unseen_weights = np.array([movie_pop_dict.get(m, 1) for m in unseen], dtype=float)
    unseen_weights = unseen_weights / unseen_weights.sum()
    
    sampled = np.random.choice(unseen, size=sample_size, replace=False, p=unseen_weights)
    
    # 해당 사용자의 가장 많은 rating_year/month/decade를 대표값으로 사용
    user_ratings = ratings[ratings['user_id'] == user]
    r_year = user_ratings['rating_year'].mode().iloc[0]
    r_month = user_ratings['rating_month'].mode().iloc[0]
    r_decade = user_ratings['rating_decade'].mode().iloc[0]
    
    negative_users.extend([user] * sample_size)
    negative_movies.extend(sampled.tolist())
    negative_labels.extend([0] * sample_size)
    negative_rating_years.extend([r_year] * sample_size)
    negative_rating_months.extend([r_month] * sample_size)
    negative_rating_decades.extend([r_decade] * sample_size)

negative_df = pd.DataFrame({
    'user_id': negative_users,
    'movie_id': negative_movies,
    'rating_year': negative_rating_years,
    'rating_month': negative_rating_months,
    'rating_decade': negative_rating_decades,
    'label': negative_labels
})
print(f"  네거티브 데이터: {negative_df.shape[0]}건")

# ============================================================
# 6. 데이터 병합
# ============================================================
print("\n[데이터 병합] 시작...")

# 선호 데이터
positive_df = positive_ratings[['user_id', 'movie_id', 'rating_year', 'rating_month', 'rating_decade', 'label']]

# 합치기
ratings_final = pd.concat([positive_df, negative_df], axis=0, ignore_index=True)

# 영화 정보 병합
movies_features = movies[['movie_id', 'movie_decade', 'movie_year', 'genre1', 'genre2', 'genre3']].copy()
merged = pd.merge(ratings_final, movies_features, on='movie_id', how='left')

# 사용자 정보 병합
users_features = users[['user_id', 'gender', 'age', 'occupation', 'zip']].copy()
merged = pd.merge(merged, users_features, on='user_id', how='left')

# [NEW] 피처 엔지니어링 피처 병합
merged = pd.merge(merged, user_stats[['user_id', 'user_avg_rating', 'user_rating_count']], on='user_id', how='left')
merged = pd.merge(merged, movie_popularity[['movie_id', 'movie_popularity_bin']], on='movie_id', how='left')

# 결측값 처리
merged.fillna('no', inplace=True)

# 최종 컬럼 순서 (기존 14개 + 신규 3개 = 17개 + label)
final_columns = [
    'user_id', 'movie_id',
    'movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade',
    'genre1', 'genre2', 'genre3',
    'gender', 'age', 'occupation', 'zip',
    'user_avg_rating', 'user_rating_count', 'movie_popularity_bin',  # NEW
    'label'
]
merged = merged[final_columns]

print(f"  최종 데이터: {merged.shape}")
print(f"  Label 분포:\n{merged['label'].value_counts()}")
print(f"  컬럼: {list(merged.columns)}")

# ============================================================
# 7. 저장
# ============================================================
merged.to_csv('ml-1m/movielens_rcmm_v3.csv', index=False)
print(f"\n✅ ml-1m/movielens_rcmm_v3.csv 저장 완료! ({merged.shape[0]} rows, {merged.shape[1]} cols)")
