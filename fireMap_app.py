import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI 산불예측 지도", layout='wide')
st.header("AI 예측 위험도 기반 산불 발생 지도 시각화")

# 1. 데이터 업로드
uploaded_file = st.file_uploader("fire.csv.xlsx 파일을 업로드하세요", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("데이터 샘플:", df.head())

    # 2. 변수명 세팅(파일에 따라 변수명을 맞추세요)
    numeric_cols = ['top_tprt', 'avg_hmd', 'ave_wdsp', 'de_rnfl_qy']
    target_col = 'frfire_ocrn_nt'

    # 3. 결측치 처리 및 타겟 이진화
    df = df.dropna(subset=[target_col])
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    df[target_col] = (df[target_col] > 0).astype(int)

    # 4. 입력/타겟 정의 (지도용 전체 데이터 학습, 더 안전하게 train_test_split 가능)
    features = numeric_cols
    X = df[features]
    y = df[target_col]

    # 5. 랜덤포레스트 학습
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 6. 예측 위험확률 계산 (지도용)
    y_proba_rf = rf.predict_proba(X)[:, 1]
    df["risk_proba"] = y_proba_rf

    # 7. 위험등급 분류
    def risk_grade(prob):
        if prob > 0.7:
            return "High"
        elif prob > 0.4:
            return "Medium"
        else:
            return "Low"
    df["risk_level"] = df["risk_proba"].apply(risk_grade)

    # 8. 색상 매핑
    risk_color = {
        "High": [255, 0, 0, 160],     # 빨강(높음)
        "Medium": [255, 165, 0, 100], # 주황(중간)
        "Low": [0, 120, 255, 60]      # 파랑(낮음)
    }
    df["risk_color"] = df["risk_level"].map(risk_color)

    # 9. 확률 슬라이더로 위험도 필터링
    min_prob, max_prob = st.slider(
        "지도에 표시할 예측 위험도(산불 발생확률) 범위 선택",
        0.0, 1.0, (0.0, 1.0), step=0.01
    )
    df_map = df[(df["risk_proba"] >= min_prob) & (df["risk_proba"] <= max_prob)]

    # 10. 지도 시각화
    if df_map.shape[0] > 0:
        lat_col, lon_col = 'lat_la', 'lon_lo'  # 위도,경도 컬럼명 확인
        midpoint = (df_map[lat_col].mean(), df_map[lon_col].mean())
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=midpoint[0],
                longitude=midpoint[1],
                zoom=6,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position=f"[{lon_col}, {lat_col}]",
                    get_color="risk_color",
                    get_radius="risk_proba * 8000 + 2000",
                    pickable=True,
                ),
            ],
            tooltip={
                "html": "위험확률: <b>{risk_proba:.2f}</b><br>위험등급: <b>{risk_level}</b>",
                "style": {"color": "white"}
            }
        ))
        st.caption("점의 색/크기: 산불 발생 AI 예측 위험확률 및 등급 (빨강=높음, 주황=중간, 파랑=낮음)")
    else:
        st.info("선택한 범위에 해당하는 데이터가 없습니다.")

    # 11. 위험등급별 건수 요약
    st.write(
        "위험등급별 산불 예측 분포 (현재 지도 표시 기준):",
        df_map["risk_level"].value_counts().to_frame("건수")
    )
else:
    st.info("먼저 데이터를 업로드하세요.")
