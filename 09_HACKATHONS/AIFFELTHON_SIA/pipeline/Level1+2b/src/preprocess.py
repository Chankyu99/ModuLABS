"""
GDELT 데이터 필터링 및 중복 제거
──────────────────────────────────────────────────────────
- 1. 국가, 이벤트 코드, 지리 정보 기반 노이즈 필터링
- 2. Actor1Name, Actor2Name 기반 중복 제거
- 3. ActionGeo_FeatureID 기반 대표 지명 통합
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.config import MONITORED_COUNTRIES, CONFIRMED_CODES, CITY_BLACKLIST, SCALING_FEATURES, SCALER_PATH

VERBOSE_LOGS = False


def _debug_print(message: str) -> None:
    if VERBOSE_LOGS:
        print(message)


def _fit_and_save_scaler(df: pd.DataFrame, scaler_file: Path) -> tuple[pd.DataFrame, StandardScaler]:
    """현재 데이터로 스케일러를 학습하고 저장한다."""
    scaler = StandardScaler()
    df[SCALING_FEATURES] = scaler.fit_transform(df[SCALING_FEATURES])
    scaler_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_file)
    return df, scaler

def unify_actiongeo_names(df: pd.DataFrame) -> pd.DataFrame:
    # Tehran/Teheran 수동 통합 로직 (FeatureID와 이름을 하나로 강제 고정)
    # 'Teheran'이든 'Tehran'이든 상관없이
    is_tehran = df['ActionGeo_FullName'].isin(['Teheran', 'Tehran'])

    if is_tehran.any():
        # 1. ID를 standard_id(보통 Tehran의 ID)로 통일
        df.loc[is_tehran, 'ActionGeo_FeatureID'] = '10074674'   # Tehran의 대표 FeatureID로 통일 (GDELT에서 가장 빈번하게 등장)
        # 2. 이름도 'Tehran'으로 통일
        df.loc[is_tehran, 'ActionGeo_FullName'] = 'Tehran'
    

    # FeatureID 기준으로 가장 자주 등장하는 지명을 대표 이름으로 선정
    name_counts = df.groupby(['ActionGeo_FeatureID', 'ActionGeo_FullName']).size().reset_index(name='count')
    best_names = name_counts.sort_values('count', ascending=False).groupby('ActionGeo_FeatureID').first()
    
    # ID당 지명이 여러 개 묶인 경우를 찾아 로그 출력
    id_name_nunique = df.groupby('ActionGeo_FeatureID')['ActionGeo_FullName'].nunique()
    inconsistent_ids = id_name_nunique[id_name_nunique > 1].index
    
    if len(inconsistent_ids) > 0:
        _debug_print("\n[전처리] 지명이 여러 개로 나타나는 데이터를 대표 지명으로 통합하였습니다.")
        for fid in inconsistent_ids:
            merged_names = df[df['ActionGeo_FeatureID'] == fid]['ActionGeo_FullName'].unique()
            primary_name = best_names.loc[fid, 'ActionGeo_FullName']
            _debug_print(f"  - ID {fid}: {list(merged_names)} -> [{primary_name}]")
            
    # 원본 데이터프레임에 대표 지명 매핑
    mapping_dict = best_names['ActionGeo_FullName'].to_dict()
    df['ActionGeo_FullName'] = df['ActionGeo_FeatureID'].map(mapping_dict)
    
    return df

def clean_gdelt_data(df: pd.DataFrame) -> pd.DataFrame:
    """GDELT 원본 데이터를 정제하고 중복을 제거하여 반환"""
    
    if df.empty:
        return pd.DataFrame()

    # 도메인 기반 필터링 (마스크 적용)
    mask = (
        (df['ActionGeo_CountryCode'].isin(MONITORED_COUNTRIES))  & 
        (df['EventCode'].isin(CONFIRMED_CODES)) & 
        (df['ActionGeo_Type'] == 4) & 
        (df['NumSources'] >= 1) &
        (~df['ActionGeo_FullName'].isin(CITY_BLACKLIST))
    )
    
    filtered = df[mask].copy()
    if filtered.empty:
        return filtered

    # Actor 수 기반 중복 제거 
    filtered['info_count'] = filtered[['Actor1Name', 'Actor2Name']].notna().sum(axis=1)
    filtered = filtered.sort_values(by='info_count', ascending=False)

    filtered = filtered.drop_duplicates(
        subset=[
            'SQLDATE', 'ActionGeo_FeatureID', 'EventCode', 
            'AvgTone', 'NumArticles', 'NumSources'
        ],
        keep='first'
    ).drop(columns=['info_count'])

    # 최종 정제된 데이터에 대해 지명 통합 수행
    filtered = unify_actiongeo_names(filtered)
    
    return filtered


def apply_standard_scaling(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """학습된 스케일러를 적용하거나, 학습 시 새로 저장"""
    if df.empty:
        return df
    
    # 1. 파생 변수 생성 (가중치 공식에 필요한 컬럼들)
    df['Log_Mentions'] = np.log1p(df['NumMentions'])
    df['Log_Sources'] = np.log1p(df['NumSources'])
    df['AvgTone_Sq'] = df['AvgTone'] ** 2
    
    scaler_file = Path(SCALER_PATH)
    
    if is_train:
        # 5개 변수(Log_Mentions, Log_Sources, AvgTone, AvgTone_Sq, GoldsteinScale)에 대해 스케일링 학습 및 적용
        df, _ = _fit_and_save_scaler(df, scaler_file)
        _debug_print(f"[전처리] 새로운 스케일러가 {scaler_file}에 저장되었습니다.")
    else:
        # 저장된 스케일러 로드하여 적용 (평균/표준편차 그대로 사용)
        if not scaler_file.exists():
            df, _ = _fit_and_save_scaler(df, scaler_file)
            _debug_print(f"[전처리] 스케일러 파일이 없어 현재 데이터로 자동 생성했습니다: {scaler_file}")
            return df
        scaler = joblib.load(scaler_file)
        df[SCALING_FEATURES] = scaler.transform(df[SCALING_FEATURES])
        
    return df
