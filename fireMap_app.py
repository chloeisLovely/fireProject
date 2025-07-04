import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

st.header("AI 예측 위험도 기반 산불 발생 지도 시각화")

# 예측 확률(산불 발생 위험도) 산출
y_proba_rf = rf.predict_proba(X)[:, 1]
df["risk_proba"] = y_proba_rf

# 위험등급 분류
def risk_grade(prob):
    if prob > 0.7:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"

df["risk_level"] = df["risk_proba"].apply(risk_grade)

# 컬러맵: 위험등급별 색상
risk_color = {
    "High": [255, 0, 0, 160],     # 빨강(높음)
    "Medium": [255, 165, 0, 100], # 주황(중간)
    "Low": [0, 120, 255, 60]      # 파랑(낮음)
}

df["risk_color"] = df["risk_level"].map(risk_color)

# 확률 슬라이더로 위험도 필터링
min_prob, max_prob = st.slider(
    "지도에 표시할 예측 위험도(산불 발생확률) 범위 선택",
    0.0, 1.0, (0.0, 1.0), step=0.01
)
df_map = df[(df["risk_proba"] >= min_prob) & (df["risk_proba"] <= max_prob)]

# 지도 중심 위치 계산
if df_map.shape[0] > 0:
    midpoint = (df_map["lat_la"].mean(), df_map["lon_lo"].mean())
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
                get_position="[lon_lo, lat_la]",
                get_color="risk_color",
                get_radius="risk_proba * 8000 + 2000",  # 위험확률 높을수록 원이 큼
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

# 위험등급별 건수 요약도 추가 출력
st.write(
    "위험등급별 산불 예측 분포 (현재 지도 표시 기준):",
    df_map["risk_level"].value_counts().to_frame("건수")
)
