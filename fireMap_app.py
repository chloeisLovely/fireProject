import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

st.set_page_config(page_title="AI 산불예측 대시보드", layout='wide')

st.title("🔥 산불 예측 AI 대시보드 (지도/모델 성능)")

uploaded_file = st.file_uploader("fire.csv.xlsx 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("데이터 샘플:", df.head())

    numeric_cols = ['top_tprt', 'avg_hmd', 'ave_wdsp', 'de_rnfl_qy']
    target_col = 'frfire_ocrn_nt'
    lat_col, lon_col = 'lat_la', 'lon_lo'

    # 결측치 및 타겟 이진화
    df = df.dropna(subset=[target_col])
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    df[target_col] = (df[target_col] > 0).astype(int)
    df = df.dropna(subset=[lat_col, lon_col])
    # 대한민국 위경도 필터
    df = df[
        (df[lat_col] > 33) & (df[lat_col] < 39) &
        (df[lon_col] > 124) & (df[lon_col] < 132)
    ]

    features = numeric_cols
    X = df[features]
    y = df[target_col]

    # 모델 학습
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)

    # 예측
    y_pred_rf = rf.predict(X)
    y_pred_lr = lr.predict(X)
    y_pred_xgb = xgb_model.predict(X)
    y_proba_rf = rf.predict_proba(X)[:, 1]
    y_proba_lr = lr.predict_proba(X)[:, 1]
    y_proba_xgb = xgb_model.predict_proba(X)[:, 1]
    df["risk_proba"] = y_proba_rf

    # 위험등급
    def risk_grade(prob):
        if prob > 0.7:
            return "High"
        elif prob > 0.4:
            return "Medium"
        else:
            return "Low"
    df["risk_level"] = df["risk_proba"].apply(risk_grade)
    risk_color = {
        "High": [255, 0, 0, 160],
        "Medium": [255, 165, 0, 100],
        "Low": [0, 120, 255, 60]
    }
    df["risk_color"] = df["risk_level"].map(risk_color)

    # 탭 만들기
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 지도 시각화", "🌳 RandomForest", "📈 LogisticRegression", "🚀 XGBoost"])

    with tab1:
        st.header("AI 예측 위험도 기반 산불 발생 지도")
        min_prob, max_prob = st.slider(
            "지도에 표시할 예측 위험도(산불 발생확률) 범위 선택",
            0.0, 1.0, (0.0, 1.0), step=0.01
        )
        df_map = df[(df["risk_proba"] >= min_prob) & (df["risk_proba"] <= max_prob)]
        st.write(f"지도 표시 대상 행 수: {df_map.shape[0]}")
        if df_map.shape[0] > 0:
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
            st.warning("지도에 표시할 데이터가 없습니다. 좌표, 필터 범위, 변수명을 다시 확인하세요.")
        st.write(
            "위험등급별 산불 예측 분포 (현재 지도 표시 기준):",
            df_map["risk_level"].value_counts().to_frame("건수")
        )

    def show_model_results(model, y_true, y_pred, y_prob, model_name, feature_importances=None):
        st.subheader(f"{model_name} - 혼동행렬")
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fire", "Fire"])
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)
        st.text(classification_report(y_true, y_pred))

        st.subheader(f"{model_name} - ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax_roc.plot([0,1],[0,1],'k--',label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f"{model_name} ROC Curve")
        ax_roc.legend()
        ax_roc.grid(True)
        st.pyplot(fig_roc)

        if feature_importances is not None:
            st.subheader(f"{model_name} - 변수 중요도")
            imp_sorted_idx = np.argsort(feature_importances)[::-1]
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(feature_importances)), feature_importances[imp_sorted_idx], align='center')
            ax_imp.set_xticks(range(len(feature_importances)))
            ax_imp.set_xticklabels(np.array(X.columns)[imp_sorted_idx], rotation=45)
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

    with tab2:
        st.header("RandomForest 결과")
        show_model_results(rf, y, y_pred_rf, y_proba_rf, "RandomForest", rf.feature_importances_)

    with tab3:
        st.header("Logistic Regression 결과")
        show_model_results(lr, y, y_pred_lr, y_proba_lr, "Logistic Regression")

    with tab4:
        st.header("XGBoost 결과")
        show_model_results(xgb_model, y, y_pred_xgb, y_proba_xgb, "XGBoost", xgb_model.feature_importances_)
else:
    st.info("먼저 데이터를 업로드하세요.")

