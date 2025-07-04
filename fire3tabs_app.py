import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

st.set_page_config(page_title="산불 예측 AI 대시보드", layout="wide")
st.title("🔥 산불 예측 AI 대시보드 (모델별 성능/시각화)")

uploaded_file = st.file_uploader("fire.csv.xlsx 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("데이터 샘플:", df.head())

    # 변수명 맞춤
    numeric_cols = ['top_tprt', 'avg_hmd', 'ave_wdsp', 'de_rnfl_qy']
    target_col = 'frfire_ocrn_nt'

    # 결측치 처리 및 타겟 이진화
    df = df.dropna(subset=[target_col])
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    df[target_col] = (df[target_col] > 0).astype(int)

    # 데이터 행수, 타겟값 종류 체크
    st.write(f"분석에 사용할 데이터 행 수: {df.shape[0]}")
    X = df[numeric_cols]
    y = df[target_col]
    n_class = y.nunique()
    st.write(f"타겟(산불발생여부) 고유값: {y.unique()} (클래스 개수: {n_class})")

    if df.shape[0] == 0 or n_class < 2:
        st.error(
            f"모델 학습/평가 불가!\n"
            f"→ 데이터가 없거나, 타겟 변수({target_col})에 0/1이 모두 있어야 합니다.\n"
            f"산불 발생(1) 데이터가 반드시 포함된 파일을 사용해 주세요."
        )
        st.stop()

    # 모델 학습
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)

    # 예측 및 예측확률
    y_pred_rf = rf.predict(X)
    y_pred_lr = lr.predict(X)
    y_pred_xgb = xgb_model.predict(X)
    y_proba_rf = rf.predict_proba(X)[:, 1]
    y_proba_lr = lr.predict_proba(X)[:, 1]
    y_proba_xgb = xgb_model.predict_proba(X)[:, 1]

    # 탭별로 모델 시각화
    tab1, tab2, tab3 = st.tabs(["🌳 RandomForest", "📈 Logistic Regression", "🚀 XGBoost"])

    def show_model_results(y_true, y_pred, y_prob, model_name, feature_names, feature_importances=None):
        # 혼동행렬
        st.subheader(f"{model_name} - 혼동행렬")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fire", "Fire"])
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)
        st.text(classification_report(y_true, y_pred))

        # ROC Curve
        st.subheader(f"{model_name} - ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f"{model_name} ROC Curve")
        ax_roc.legend()
        ax_roc.grid(True)
        st.pyplot(fig_roc)

        # 변수 중요도
        if feature_importances is not None:
            st.subheader(f"{model_name} - 변수 중요도")
            imp_sorted_idx = np.argsort(feature_importances)[::-1]
            fig_imp, ax_imp = plt.subplots()
            ax_imp.bar(range(len(feature_importances)), feature_importances[imp_sorted_idx], align='center')
            ax_imp.set_xticks(range(len(feature_importances)))
            ax_imp.set_xticklabels(np.array(feature_names)[imp_sorted_idx], rotation=45)
            ax_imp.set_title("Feature Importances")
            st.pyplot(fig_imp)

    with tab1:
        st.header("RandomForest 결과")
        show_model_results(
            y, y_pred_rf, y_proba_rf, "RandomForest", numeric_cols, rf.feature_importances_
        )

    with tab2:
        st.header("Logistic Regression 결과")
        show_model_results(
            y, y_pred_lr, y_proba_lr, "Logistic Regression", numeric_cols
        )

    with tab3:
        st.header("XGBoost 결과")
        show_model_results(
            y, y_pred_xgb, y_proba_xgb, "XGBoost", numeric_cols, xgb_model.feature_importances_
        )

else:
    st.info("먼저 데이터를 업로드하세요.")
