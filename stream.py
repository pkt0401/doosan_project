import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 언어 설정 텍스트 정의
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_phase1": "위험성 평가 (Phase 1)",
        "tab_phase2": "개선대책 생성 (Phase 2)",
        "tab_history": "평가 이력",
        "tab_statistics": "통계 분석",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": """
        LLM(Large Language Model)을 활용한 위험성평가 자동화 시스템은 건설 현장의 안전 관리를 혁신적으로 개선합니다:
        
        1. <span class="highlight">작업 내용 입력 시 생성형 AI를 통한 '유해위험요인' 자동 예측 및 위험 등급 산정</span> <span class="phase-badge">Phase 1</span>
        2. <span class="highlight">위험도 감소를 위한 개선대책 자동 생성 및 감소율 예측</span> <span class="phase-badge">Phase 2</span>
        3. AI는 건설현장의 기존 위험성평가를 공정별로 구분하고, 해당 유해위험요인을 학습
        4. 자동 생성 기술 개발 완료 후 위험도 기반 사고위험성 분석 및 개선대책 생성
        
        이 시스템은 PIMS 및 안전지킴이 등 EHS 플랫폼에 AI 기술 탑재를 통해 통합 사고 예측 프로그램으로 발전 예정입니다.
        """,
        "process_title": "AI 위험성평가 프로세스",
        "process_steps": ["작업내용 입력", "AI 위험분석", "유해요인 예측", "위험등급 산정", "개선대책 자동생성", "안전조치 적용"],
        "features_title": "시스템 특징 및 구성요소",
        "phase1_features": """
        #### Phase 1: 위험성 평가 자동화
        - 공정별 작업활동에 따른 위험성평가 데이터 학습
        - 작업활동 입력 시 유해위험요인 자동 예측
        - 유사 위험요인 사례 검색 및 표시
        - 대규모 언어 모델(LLM) 기반 위험도(빈도, 강도, T) 측정
        - Excel 기반 공정별 위험성평가 데이터 자동 분석
        - 위험등급(A-E) 자동 산정
        """,
        "phase2_features": """
        #### Phase 2: 개선대책 자동 생성
        - 위험요소별 맞춤형 개선대책 자동 생성
        - 다국어(한/영/중) 개선대책 생성 지원
        - 개선 전후 위험도(T) 자동 비교 분석
        - 위험 감소율(RRR) 정량적 산출
        - 공종/공정별 최적 개선대책 데이터베이스 구축
        """,
        "phase1_header": "위험성 평가 자동화 (Phase 1)",
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택",
        "load_data_label": "데이터 로드 및 인덱스 구성",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "현재 {total_rows}개의 데이터를 처리합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {total_rows}개 항목 처리)",
        "hazard_prediction_header": "유해위험요인 예측",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "predict_hazard_btn": "유해위험요인 예측하기",
        "activity_warning": "작업활동을 입력하세요.",
        "predicting_hazard": "유해위험요인을 예측하는 중...",
        "similar_cases_header": "유사한 사례",
        "similar_case_text": """
        <div class="similar-case">
            <strong>사례 {i}</strong><br>
            <strong>작업활동:</strong> {activity}<br>
            <strong>유해위험요인:</strong> {hazard}<br>
            <strong>위험도:</strong> 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})
        </div>
        """,
        "prediction_result_header": "예측 결과",
        "activity_result": "작업활동: {activity}",
        "hazard_result": "예측된 유해위험요인: {hazard}",
        "result_table_columns": ["항목", "값"],
        "result_table_rows": ["빈도", "강도", "T 값", "위험등급"],
        "parsing_error": "위험성 평가 결과를 파싱할 수 없습니다.",
        "gpt_response": "GPT 원문 응답: {response}",
        "phase2_header": "개선대책 자동 생성 (Phase 2)",
        "language_select_label": "개선대책 언어 선택:",
        "input_method_label": "입력 방식 선택:",
        "input_methods": ["Phase 1 평가 결과 사용", "직접 입력"],
        "phase1_results_header": "Phase 1 평가 결과",
        "risk_level_text": "위험도: 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})",
        "phase1_first_warning": "먼저 Phase 1에서 위험성 평가를 수행하세요.",
        "hazard_label": "유해위험요인:",
        "frequency_label": "빈도 (1-5):",
        "intensity_label": "강도 (1-5):",
        "t_value_text": "T값: {t_value} (등급: {grade})",
        "generate_improvement_btn": "개선대책 생성",
        "generating_improvement": "개선대책을 생성하는 중...",
        "no_data_warning": "Phase 1에서 데이터 로드 및 인덱스 구성을 완료하지 않았습니다. 기본 예시를 사용합니다.",
        "improvement_result_header": "개선대책 생성 결과",
        "improvement_plan_header": "개선대책",
        "risk_improvement_header": "위험도 개선 결과",
        "comparison_columns": ["항목", "개선 전", "개선 후"],
        "risk_reduction_label": "위험 감소율 (RRR)",
        "t_value_change_header": "위험도(T값) 변화",
        "before_improvement": "개선 전 T값:",
        "after_improvement": "개선 후 T값:",
        "parsing_error_improvement": "개선대책 생성 결과를 파싱할 수 없습니다.",
        "save_assessment": "평가 결과 저장",
        "assessment_saved": "평가 결과가 저장되었습니다.",
        "export_excel": "Excel로 내보내기",
        "export_pdf": "PDF로 내보내기",
        "history_header": "평가 이력",
        "statistics_header": "통계 분석",
        "risk_distribution": "위험등급 분포",
        "monthly_trend": "월별 평가 추이",
        "work_type_analysis": "작업유형별 위험도 분석",
        "confidence_score": "신뢰도 점수: {score}%",
        "data_insights": "데이터 인사이트",
        "total_assessments": "총 평가 건수",
        "high_risk_count": "고위험 (A등급) 건수",
        "avg_risk_score": "평균 위험도(T값)",
        "improvement_rate": "개선율"
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_phase1": "Risk Assessment (Phase 1)",
        "tab_phase2": "Improvement Measures (Phase 2)",
        "tab_history": "Assessment History",
        "tab_statistics": "Statistical Analysis",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": """
        The risk assessment automation system using LLM (Large Language Model) innovatively improves safety management at construction sites:
        
        1. <span class="highlight">Automatic prediction of 'hazards' and risk level calculation through generative AI</span> <span class="phase-badge">Phase 1</span>
        2. <span class="highlight">Automatic generation of improvement measures and reduction rate prediction to reduce risk level</span> <span class="phase-badge">Phase 2</span>
        3. AI learns existing risk assessments at construction sites by process and their hazard factors
        4. After the development of automatic generation technology, risk analysis and improvement measures based on risk level
        
        This system is expected to evolve into an integrated accident prediction program through the incorporation of AI technology into EHS platforms such as PIMS and Safety Guardian.
        """,
        # ... (영어 텍스트는 기존과 동일하므로 생략)
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_phase1": "风险评估 (第1阶段)",
        "tab_phase2": "改进措施 (第2阶段)",
        "tab_history": "评估历史",
        "tab_statistics": "统计分析",
        # ... (중국어 텍스트는 기존과 동일하므로 생략)
    }
}

# 페이지 설정
st.set_page_config(
    page_title="Artificial Intelligence Risk Assessment",
    page_icon="🛠️",
    layout="wide"
)

# 개선된 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8eaf6 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .highlight {
        background: linear-gradient(120deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 500;
    }
    .result-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .phase-badge {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 10px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(76, 175, 80, 0.3);
    }
    .similar-case {
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
        border-left: 4px solid #689f38;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .language-selector {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #155724;
    }
    .warning-message {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    .confidence-badge {
        background: linear-gradient(45deg, #2196F3, #1976D2);
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
    }
    .data-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        border: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976D2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 (개선된 버전)
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "retriever_pool_df" not in st.session_state:
    st.session_state.retriever_pool_df = None
if "language" not in st.session_state:
    st.session_state.language = "Korean"
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []
if "last_assessment" not in st.session_state:
    st.session_state.last_assessment = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None

# 상단에 언어 선택기 추가
col1, col2 = st.columns([6, 1])
with col2:
    selected_language = st.selectbox(
        "",
        options=list(system_texts.keys()),
        index=list(system_texts.keys()).index(st.session_state.language) if st.session_state.language in system_texts else 0,
        key="language_selector"
    )
    st.session_state.language = selected_language

# 현재 언어에 따른 텍스트 가져오기
texts = system_texts[st.session_state.language]

# 헤더 표시
st.markdown(f'<div class="main-header">{texts["title"]}</div>', unsafe_allow_html=True)

# 탭 설정 (개선된 버전 - 이력 및 통계 탭 추가)
tabs = st.tabs([
    texts["tab_overview"], 
    texts["tab_phase1"], 
    texts["tab_phase2"],
    texts.get("tab_history", "Assessment History"),
    texts.get("tab_statistics", "Statistical Analysis")
])

# ------------------ 유틸리티 함수 (개선된 버전) ------------------

def determine_grade(value):
    """빈도*강도 결과 T에 따른 등급 결정 함수."""
    if 16 <= value <= 25:
        return 'A'
    elif 10 <= value <= 15:
        return 'B'
    elif 5 <= value <= 9:
        return 'C'
    elif 3 <= value <= 4:
        return 'D'
    elif 1 <= value <= 2:
        return 'E'
    else:
        return '알 수 없음' if st.session_state.language == 'Korean' else 'Unknown'

def calculate_confidence_score(retrieved_docs, similarity_scores=None):
    """신뢰도 점수 계산"""
    if similarity_scores is not None:
        # 유사도 점수 기반 신뢰도 계산
        avg_similarity = np.mean(similarity_scores) if len(similarity_scores) > 0 else 0.5
        confidence = min(100, avg_similarity * 100)
    else:
        # 검색된 문서 수와 데이터 품질 기반 추정
        doc_count = len(retrieved_docs)
        if doc_count >= 3:
            confidence = 85
        elif doc_count >= 2:
            confidence = 75
        else:
            confidence = 65
    
    return round(confidence, 1)

def load_data(selected_dataset_name):
    """개선된 데이터 로드 함수 - 더 많은 필드 처리"""
    try:
        # 실제 Excel 파일에서 데이터 로드
        df = pd.read_excel(f"{selected_dataset_name}.xlsx")
        
        # 데이터 전처리 개선
        # 헤더 정리
        df.columns = df.columns.str.strip()
        
        # 불필요한 컬럼 제거
        columns_to_drop = ['삭제 Del', 'Unnamed: 0'] if any(col in df.columns for col in ['삭제 Del', 'Unnamed: 0']) else []
        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
        
        # 첫 번째 행이 헤더인 경우 처리
        if df.iloc[0].isna().sum() > len(df.columns) * 0.5:
            df = df.iloc[1:]
        
        # 컬럼명 표준화
        column_mapping = {
            '작업활동 및 내용\nWork & Contents': '작업활동 및 내용',
            '유해위험요인 및 환경측면 영향\nHazard & Risk': '유해위험요인 및 환경측면 영향',
            '피해형태 및 환경영향\nDamage & Effect': '피해형태 및 환경영향'
        }
        df = df.rename(columns=column_mapping)
        
        # 빈도, 강도 컬럼 찾기 및 이름 변경
        freq_cols = [col for col in df.columns if '빈도' in str(col) or 'Frequency' in str(col)]
        intensity_cols = [col for col in df.columns if '강도' in str(col) or 'Severity' in str(col) or 'Intensity' in str(col)]
        
        if freq_cols:
            df = df.rename(columns={freq_cols[0]: '빈도'})
        if intensity_cols:
            df = df.rename(columns={intensity_cols[0]: '강도'})
        
        # 숫자형 변환 및 T값 계산
        if '빈도' in df.columns and '강도' in df.columns:
            df['빈도'] = pd.to_numeric(df['빈도'], errors='coerce').fillna(3)
            df['강도'] = pd.to_numeric(df['강도'], errors='coerce').fillna(3)
            df['T'] = df['빈도'] * df['강도']
        else:
            # 기본값 설정
            df['빈도'] = 3
            df['강도'] = 3
            df['T'] = 9
        
        # 등급 계산
        df['등급'] = df['T'].apply(determine_grade)
        
        # 빈 행 제거
        df = df.dropna(subset=['작업활동 및 내용', '유해위험요인 및 환경측면 영향'], how='all')
        
        # 개선대책 컬럼 확인
        improvement_cols = [col for col in df.columns if any(keyword in str(col) for keyword in ['개선', 'Improvement', 'Corrective', '대책'])]
        if improvement_cols:
            df['개선대책'] = df[improvement_cols[0]]
        
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        
        # 더 포괄적인 샘플 데이터 생성
        st.warning("Excel 파일을 찾을 수 없어 확장된 샘플 데이터를 생성합니다.")
        
        sample_data = {
            "작업활동 및 내용": [
                "Unloading of steel structure materials using forklift at temporary site storage area",
                "Installation of Concrete / CMU blocks", 
                "Excavation and backfill work",
                "Steel reinforcement installation",
                "Concrete pouring and finishing",
                "Scaffolding assembly and dismantling",
                "Electrical wiring and installation",
                "Welding operations",
                "Heavy equipment operation",
                "Material handling and transportation"
            ],
            "유해위험요인 및 환경측면 영향": [
                "Fall of load due to multiple lifting",
                "Fall due to insufficient working platform",
                "Cave-in due to unstable soil conditions",
                "Cut injury from rebar handling",
                "Chemical exposure from concrete additives",
                "Fall from height during assembly",
                "Electric shock from live wires",
                "Fire and explosion from welding",
                "Struck by moving equipment",
                "Musculoskeletal injury from manual handling"
            ],
            "피해형태 및 환경영향": [
                "Injury from falling objects",
                "Fall injury",
                "Burial and crushing",
                "Laceration",
                "Chemical burns",
                "Fall from height",
                "Electrocution",
                "Burns and fire",
                "Impact injury",
                "Strain and sprain"
            ],
            "빈도": [3, 3, 4, 2, 2, 4, 3, 2, 3, 4],
            "강도": [5, 5, 4, 3, 4, 5, 5, 4, 4, 2],
            "개선대책": [
                "1) Install proper rigging equipment 2) Conduct pre-lift safety checks 3) Maintain clear communication",
                "1) Install missing scaffold planks 2) Use full body harness 3) Install safety railings",
                "1) Proper soil analysis 2) Install shoring system 3) Regular inspection",
                "1) Use proper PPE 2) Safe handling procedures 3) Tool maintenance",
                "1) Proper ventilation 2) Use appropriate PPE 3) Material safety procedures",
                "1) Competent person supervision 2) Fall protection system 3) Regular inspection",
                "1) LOTO procedures 2) Qualified electrician 3) Proper insulation",
                "1) Fire watch personnel 2) Proper ventilation 3) Hot work permits",
                "1) Designated traffic routes 2) Spotters 3) Equipment maintenance",
                "1) Mechanical aids 2) Proper lifting techniques 3) Team lifting"
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df['T'] = df['빈도'] * df['강도']
        df['등급'] = df['T'].apply(determine_grade)
        
        return df

def embed_texts_with_openai(texts, model="text-embedding-3-large", api_key=None):
    """개선된 OpenAI 임베딩 생성 함수"""
    if api_key:
        openai.api_key = api_key
    
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)
    
    # 배치 처리로 효율성 개선
    batch_size = 10
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            # 텍스트 전처리
            processed_texts = [str(text).replace("\n", " ").strip() for text in batch_texts]
            
            response = openai.Embedding.create(
                model=model, 
                input=processed_texts
            )
            
            batch_embeddings = [item["embedding"] for item in response["data"]]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            st.error(f"배치 {i//batch_size + 1} 임베딩 중 오류: {str(e)}")
            # 오류 발생 시 영벡터로 대체
            for _ in batch_texts:
                embeddings.append([0.0] * 1536)
        
        progress_bar.progress(min(1.0, (i + batch_size) / total))
    
    return embeddings

def generate_with_gpt(prompt, api_key=None, model="gpt-4o-mini", language="Korean"):
    """개선된 GPT 호출 함수"""
    if api_key:
        openai.api_key = api_key
    
    system_prompts = {
        "Korean": "위험성 평가 및 개선대책 생성을 돕는 전문 도우미입니다. 정확하고 구체적인 안전 관리 조치를 제안합니다.",
        "English": "I am a professional assistant helping with risk assessment and improvement measures. I provide accurate and specific safety management recommendations.",
        "Chinese": "我是一个协助进行风险评估和改进措施的专业助手。我提供准确和具体的安全管理建议。"
    }
    
    system_prompt = system_prompts.get(language, system_prompts["Korean"])
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 더 일관된 결과를 위해 낮춤
            max_tokens=500,   # 더 상세한 응답을 위해 증가
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return None

def save_assessment_to_history(assessment_data):
    """평가 결과를 이력에 저장"""
    assessment_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assessment_data['id'] = len(st.session_state.assessment_history) + 1
    st.session_state.assessment_history.append(assessment_data)

def export_to_excel(data, filename="risk_assessment_results.xlsx"):
    """Excel 파일로 내보내기"""
    try:
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False, engine='openpyxl')
        return filename
    except Exception as e:
        st.error(f"Excel 내보내기 오류: {str(e)}")
        return None

def create_risk_visualization(assessment_history):
    """위험도 시각화 차트 생성"""
    if not assessment_history:
        return None
    
    df = pd.DataFrame(assessment_history)
    
    # 위험등급 분포 차트
    grade_counts = df['grade'].value_counts()
    fig_grade = px.pie(
        values=grade_counts.values, 
        names=grade_counts.index,
        title="위험등급 분포",
        color_discrete_map={
            'A': '#FF4444', 'B': '#FF8800', 'C': '#FFCC00', 
            'D': '#88CC00', 'E': '#44CC44'
        }
    )
    
    # 월별 추이 차트
    df['date'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    
    fig_trend = px.line(
        x=monthly_counts.index.astype(str), 
        y=monthly_counts.values,
        title="월별 평가 건수 추이",
        labels={'x': '월', 'y': '평가 건수'}
    )
    
    return fig_grade, fig_trend

# Phase 1 관련 함수들
def construct_prompt_phase1_hazard(retrieved_docs, activity_text, language="Korean"):
    """작업활동으로부터 유해위험요인을 예측하는 프롬프트 생성 (개선된 버전)"""
    prompt_templates = {
        "Korean": {
            "intro": "다음은 건설 현장의 작업활동과 그에 따른 유해위험요인의 예시입니다:\n\n",
            "example_format": "예시 {i}:\n작업활동: {activity}\n유해위험요인: {hazard}\n위험도: T={t_value} (등급 {grade})\n\n",
            "query_format": "이제 다음 작업활동에 대한 유해위험요인을 예측해주세요. 구체적이고 실무적인 위험요인을 제시하세요:\n작업활동: {activity}\n\n예측된 유해위험요인: "
        },
        "English": {
            "intro": "The following are examples of work activities at construction sites and their associated hazards:\n\n",
            "example_format": "Example {i}:\nWork Activity: {activity}\nHazard: {hazard}\nRisk Level: T={t_value} (Grade {grade})\n\n",
            "query_format": "Now, please predict the hazard for the following work activity. Provide specific and practical risk factors:\nWork Activity: {activity}\n\nPredicted Hazard: "
        },
        "Chinese": {
            "intro": "以下是建筑工地的工作活动及其相关危害的例子:\n\n",
            "example_format": "例子 {i}:\n工作活动: {activity}\n危害: {hazard}\n风险等级: T={t_value} (等级 {grade})\n\n",
            "query_format": "现在，请预测以下工作活动的危害。请提供具体和实用的风险因素:\n工作活动: {activity}\n\n预测的危害: "
        }
    }
    
    template = prompt_templates.get(language, prompt_templates["Korean"])
    
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            activity = doc['작업활동 및 내용']
            hazard = doc['유해위험요인 및 환경측면 영향']
            t_value = doc.get('T', 0)
            grade = doc.get('등급', 'C')
            retrieved_examples.append((activity, hazard, t_value, grade))
        except:
            continue
    
    prompt = template["intro"]
    for i, (activity, hazard, t_value, grade) in enumerate(retrieved_examples, 1):
        prompt += template["example_format"].format(
            i=i, activity=activity, hazard=hazard, t_value=t_value, grade=grade
        )
    
    prompt += template["query_format"].format(activity=activity_text)
    
    return prompt

def construct_prompt_phase1_risk(retrieved_docs, activity_text, hazard_text, language="Korean"):
    """빈도와 강도 예측을 위한 개선된 프롬프트"""
    prompt_templates = {
        "Korean": {
            "intro": "다음은 작업활동과 유해위험요인에 따른 위험도 평가 예시입니다:\n\n",
            "example_format": "예시 {i}:\n작업활동: {activity}\n유해위험요인: {hazard}\n빈도: {freq} (1=매우 드뭄, 2=드뭄, 3=보통, 4=자주, 5=매우 자주)\n강도: {intensity} (1=경미, 2=약간, 3=보통, 4=심각, 5=치명적)\nT값: {t_value}\n\n",
            "query_format": "다음 작업활동과 유해위험요인에 대해 빈도(1-5)와 강도(1-5)를 평가하세요:\n\n작업활동: {activity}\n유해위험요인: {hazard}\n\n평가 기준:\n- 빈도: 해당 위험이 발생할 가능성 (1=매우 드뭄 ~ 5=매우 자주)\n- 강도: 사고 발생 시 피해 정도 (1=경미 ~ 5=치명적)\n\n다음 JSON 형식으로 정확히 응답하세요:\n{json_format}\n\n응답:"
        },
        "English": {
            "intro": "The following are examples of risk assessment based on work activities and hazards:\n\n",
            "example_format": "Example {i}:\nWork Activity: {activity}\nHazard: {hazard}\nFrequency: {freq} (1=Very Rare, 2=Rare, 3=Moderate, 4=Frequent, 5=Very Frequent)\nSeverity: {intensity} (1=Minor, 2=Slight, 3=Moderate, 4=Serious, 5=Fatal)\nT-value: {t_value}\n\n",
            "query_format": "Please evaluate the frequency (1-5) and severity (1-5) for the following work activity and hazard:\n\nWork Activity: {activity}\nHazard: {hazard}\n\nEvaluation Criteria:\n- Frequency: Likelihood of the risk occurring (1=Very Rare ~ 5=Very Frequent)\n- Severity: Degree of harm if accident occurs (1=Minor ~ 5=Fatal)\n\nPlease respond exactly in the following JSON format:\n{json_format}\n\nResponse:"
        },
        "Chinese": {
            "intro": "以下是基于工作活动和危害的风险评估示例:\n\n",
            "example_format": "示例 {i}:\n工作活动: {activity}\n危害: {hazard}\n频率: {freq} (1=非常罕见, 2=罕见, 3=中等, 4=频繁, 5=非常频繁)\n严重程度: {intensity} (1=轻微, 2=轻度, 3=中等, 4=严重, 5=致命)\nT值: {t_value}\n\n",
            "query_format": "请评估以下工作活动和危害的频率(1-5)和严重程度(1-5):\n\n工作活动: {activity}\n危害: {hazard}\n\n评估标准:\n- 频率: 风险发生的可能性 (1=非常罕见 ~ 5=非常频繁)\n- 严重程度: 事故发生时的伤害程度 (1=轻微 ~ 5=致命)\n\n请完全按照以下JSON格式回答:\n{json_format}\n\n回答:"
        }
    }
    
    json_formats = {
        "Korean": '{"빈도": 숫자, "강도": 숫자, "T": 숫자}',
        "English": '{"frequency": number, "intensity": number, "T": number}',
        "Chinese": '{"频率": 数字, "强度": 数字, "T": 数字}'
    }
    
    template = prompt_templates.get(language, prompt_templates["Korean"])
    json_format = json_formats.get(language, json_formats["Korean"])
    
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            activity = doc['작업활동 및 내용']
            hazard = doc['유해위험요인 및 환경측면 영향']
            frequency = int(doc['빈도'])
            intensity = int(doc['강도'])
            T_value = frequency * intensity
            retrieved_examples.append((activity, hazard, frequency, intensity, T_value))
        except:
            continue
    
    prompt = template["intro"]
    for i, (activity, hazard, freq, intensity, t_value) in enumerate(retrieved_examples[:3], 1):
        prompt += template["example_format"].format(
            i=i, activity=activity, hazard=hazard, 
            freq=freq, intensity=intensity, t_value=t_value
        )
    
    prompt += template["query_format"].format(
        activity=activity_text, 
        hazard=hazard_text,
        json_format=json_format
    )
    
    return prompt

def parse_gpt_output_phase1(gpt_output, language="Korean"):
    """GPT 출력 파싱 개선 버전"""
    json_patterns = {
        "Korean": r'\{"빈도":\s*([1-5]),\s*"강도":\s*([1-5]),\s*"T":\s*([0-9]+)\}',
        "English": r'\{"frequency":\s*([1-5]),\s*"intensity":\s*([1-5]),\s*"T":\s*([0-9]+)\}',
        "Chinese": r'\{"频率":\s*([1-5]),\s*"强度":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    }
    
    # 우선 현재 언어 패턴으로 시도
    pattern = json_patterns.get(language, json_patterns["Korean"])
    match = re.search(pattern, gpt_output)
    
    if match:
        pred_frequency = int(match.group(1))
        pred_intensity = int(match.group(2))
        pred_T = int(match.group(3))
        return pred_frequency, pred_intensity, pred_T
    
    # 다른 언어 패턴들도 시도
    for lang, pattern in json_patterns.items():
        if lang != language:
            match = re.search(pattern, gpt_output)
            if match:
                pred_frequency = int(match.group(1))
                pred_intensity = int(match.group(2))
                pred_T = int(match.group(3))
                return pred_frequency, pred_intensity, pred_T
    
    # 패턴 매칭 실패 시 숫자만 추출 시도
    numbers = re.findall(r'\b([1-5])\b', gpt_output)
    if len(numbers) >= 2:
        pred_frequency = int(numbers[0])
        pred_intensity = int(numbers[1])
        pred_T = pred_frequency * pred_intensity
        return pred_frequency, pred_intensity, pred_T
    
    return None

# Phase 2 관련 함수들
def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    """개선대책 생성을 위한 개선된 프롬프트"""
    
    example_section = ""
    examples_added = 0
    
    field_names = {
        "Korean": {
            "improvement_fields": ['개선대책', '개선대책 및 세부관리방안', '개선방안', 'Corrective Action'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향"
        },
        "English": {
            "improvement_fields": ['Improvement Measures', 'Improvement Plan', 'Countermeasures', '개선대책'],
            "activity": "작업활동 및 내용", 
            "hazard": "유해위험요인 및 환경측면 영향"
        },
        "Chinese": {
            "improvement_fields": ['改进措施', '改进计划', '对策', '개선대책'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향"
        }
    }
    
    fields = field_names.get(target_language, field_names["Korean"])
    
    # 검색된 문서에서 예시 생성
    for _, row in retrieved_docs.iterrows():
        try:
            improvement_plan = ""
            for field in fields["improvement_fields"]:
                if field in row and pd.notna(row[field]) and str(row[field]).strip():
                    improvement_plan = str(row[field]).strip()
                    break
            
            if not improvement_plan:
                continue
                
            original_freq = int(row['빈도']) if '빈도' in row else 3
            original_intensity = int(row['강도']) if '강도' in row else 3
            original_T = original_freq * original_intensity
            
            # 개선 후 값 추정 (실제 데이터가 없는 경우)
            improved_freq = max(1, original_freq - 1)
            improved_intensity = max(1, original_intensity - 1)
            improved_T = improved_freq * improved_intensity
            
            example_section += f"""
예시 {examples_added + 1}:
작업활동: {row[fields['activity']]}
유해위험요인: {row[fields['hazard']]}
원래 위험도: 빈도 {original_freq}, 강도 {original_intensity}, T={original_T}
개선대책: {improvement_plan}
개선 후 위험도: 빈도 {improved_freq}, 강도 {improved_intensity}, T={improved_T}
위험 감소율: {((original_T - improved_T) / original_T * 100):.1f}%

"""
            examples_added += 1
            if examples_added >= 2:
                break
                
        except Exception as e:
            continue
    
    # 기본 예시 (언어별)
    if examples_added == 0:
        if target_language == "Korean":
            example_section = """
예시 1:
작업활동: 굴착 및 되메우기 작업
유해위험요인: 부적절한 경사로 인한 굴착벽 붕괴
원래 위험도: 빈도 3, 강도 4, T=12
개선대책: 1) 토양 분류에 따른 적절한 경사 유지 2) 굴착 벽면 보강 시설 설치 3) 정기적인 지반 상태 점검 실시 4) 작업자 안전교육 강화
개선 후 위험도: 빈도 1, 강도 2, T=2
위험 감소율: 83.3%

예시 2:
작업활동: 중장비를 이용한 자재 운반
유해위험요인: 운반 중 자재 낙하로 인한 충돌
원래 위험도: 빈도 2, 강도 5, T=10
개선대책: 1) 적절한 리깅 장비 사용 2) 작업 전 안전점검 실시 3) 신호수 배치 4) 안전구역 설정 및 출입통제
개선 후 위험도: 빈도 1, 강도 2, T=2
위험 감소율: 80.0%

"""
        elif target_language == "English":
            example_section = """
Example 1:
Work Activity: Excavation and backfilling
Hazard: Collapse of excavation wall due to improper sloping
Original Risk: Frequency 3, Intensity 4, T=12
Improvement Measures: 1) Maintain proper slope according to soil classification 2) Install excavation wall reinforcement 3) Conduct regular ground condition inspections 4) Enhance worker safety training
Improved Risk: Frequency 1, Intensity 2, T=2
Risk Reduction Rate: 83.3%

Example 2:
Work Activity: Material transportation using heavy equipment
Hazard: Material fall causing collision during transport
Original Risk: Frequency 2, Intensity 5, T=10
Improvement Measures: 1) Use appropriate rigging equipment 2) Conduct pre-work safety inspections 3) Deploy signal personnel 4) Establish safety zones and access control
Improved Risk: Frequency 1, Intensity 2, T=2
Risk Reduction Rate: 80.0%

"""
    
    # 언어별 JSON 키와 지시사항
    json_keys = {
        "Korean": {
            "improvement": "개선대책",
            "improved_freq": "개선_후_빈도", 
            "improved_intensity": "개선_후_강도",
            "improved_t": "개선_후_T",
            "reduction_rate": "위험_감소율"
        },
        "English": {
            "improvement": "improvement_measures",
            "improved_freq": "improved_frequency",
            "improved_intensity": "improved_intensity", 
            "improved_t": "improved_T",
            "reduction_rate": "risk_reduction_rate"
        },
        "Chinese": {
            "improvement": "改进措施",
            "improved_freq": "改进后频率",
            "improved_intensity": "改进后强度",
            "improved_t": "改进后T值", 
            "reduction_rate": "风险降低率"
        }
    }
    
    instructions = {
        "Korean": {
            "task": "다음 작업활동과 유해위험요인에 대한 구체적이고 실행 가능한 개선대책을 제시하고, 개선 후 위험도를 평가하세요:",
            "guidelines": """
개선대책 작성 가이드라인:
- 최소 4개 이상의 구체적인 개선조치를 제시하세요
- 기술적 대책, 관리적 대책, 개인보호구 대책을 균형있게 포함하세요
- 실제 현장에서 적용 가능한 현실적인 방안을 제시하세요
- 각 대책은 번호를 매겨 명확히 구분하세요

위험도 평가 기준:
- 개선 후 빈도는 원래 빈도보다 1-2단계 낮게 평가
- 개선 후 강도는 대책의 효과성에 따라 조정
- 현실적인 개선 효과를 반영하세요""",
            "output_instruction": "다음 JSON 형식으로 정확히 응답하세요:"
        },
        "English": {
            "task": "Provide specific and actionable improvement measures for the following work activity and hazard, and evaluate the post-improvement risk level:",
            "guidelines": """
Improvement Measures Guidelines:
- Provide at least 4 specific improvement actions
- Include a balanced mix of technical, administrative, and PPE measures
- Suggest realistic solutions applicable in actual field conditions
- Clearly distinguish each measure with numbering

Risk Assessment Criteria:
- Post-improvement frequency should be 1-2 levels lower than original
- Post-improvement intensity should be adjusted based on measure effectiveness
- Reflect realistic improvement effects""",
            "output_instruction": "Please respond exactly in the following JSON format:"
        },
        "Chinese": {
            "task": "为以下工作活动和危害提供具体可行的改进措施，并评估改进后的风险等级：",
            "guidelines": """
改进措施指导原则:
- 提供至少4项具体的改进行动
- 包括技术措施、管理措施和个人防护设备措施的平衡组合
- 建议在实际现场条件下可应用的现实解决方案
- 用编号清晰区分每项措施

风险评估标准:
- 改进后频率应比原始频率低1-2个等级
- 改进后强度应根据措施有效性进行调整
- 反映现实的改进效果""",
            "output_instruction": "请完全按照以下JSON格式回答："
        }
    }
    
    keys = json_keys.get(target_language, json_keys["Korean"])
    instr = instructions.get(target_language, instructions["Korean"])
    
    # 최종 프롬프트 구성
    prompt = f"""{example_section}

{instr['task']}

작업활동: {activity_text}
유해위험요인: {hazard_text}
현재 위험도: 빈도 {freq}, 강도 {intensity}, T={T}

{instr['guidelines']}

{instr['output_instruction']}
{{
    "{keys['improvement']}": "구체적인 개선대책 목록 (최소 4개 항목)",
    "{keys['improved_freq']}": 숫자 (1-5),
    "{keys['improved_intensity']}": 숫자 (1-5),
    "{keys['improved_t']}": 숫자,
    "{keys['reduction_rate']}": 숫자 (백분율)
}}

응답:"""
    
    return prompt

def parse_gpt_output_phase2(gpt_output, language="Korean"):
    """개선된 Phase 2 GPT 출력 파싱"""
    try:
        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', gpt_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 블록 표시가 없는 경우 중괄호 내용 추출
            brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', gpt_output, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)
            else:
                json_str = gpt_output.strip()
        
        # JSON 파싱
        result = json.loads(json_str)
        
        # 언어별 키 매핑
        key_mappings = {
            "Korean": {
                "improvement": ["개선대책", "개선방안", "개선조치"],
                "improved_freq": ["개선_후_빈도", "개선후빈도", "개선 후 빈도"],
                "improved_intensity": ["개선_후_강도", "개선후강도", "개선 후 강도"],
                "improved_t": ["개선_후_T", "개선후T", "개선 후 T"],
                "reduction_rate": ["위험_감소율", "감소율", "위험감소율"]
            },
            "English": {
                "improvement": ["improvement_measures", "improvement_plan", "improvements"],
                "improved_freq": ["improved_frequency", "new_frequency"],
                "improved_intensity": ["improved_intensity", "new_intensity"],
                "improved_t": ["improved_T", "new_T"],
                "reduction_rate": ["risk_reduction_rate", "reduction_rate"]
            },
            "Chinese": {
                "improvement": ["改进措施", "改进计划"],
                "improved_freq": ["改进后频率", "新频率"],
                "improved_intensity": ["改进后强度", "新强度"],
                "improved_t": ["改进后T值", "新T值"],
                "reduction_rate": ["风险降低率", "降低率"]
            }
        }
        
        # 결과 매핑
        mapped_result = {}
        mappings = key_mappings.get(language, key_mappings["Korean"])
        
        for result_key, possible_keys in mappings.items():
            for key in possible_keys:
                if key in result:
                    mapped_result[result_key] = result[key]
                    break
            # 키를 찾지 못한 경우 기본값 설정
            if result_key not in mapped_result:
                if result_key == "improved_freq":
                    mapped_result[result_key] = 2
                elif result_key == "improved_intensity":
                    mapped_result[result_key] = 2
                elif result_key == "improved_t":
                    mapped_result[result_key] = 4
                elif result_key == "reduction_rate":
                    mapped_result[result_key] = 50.0
                elif result_key == "improvement":
                    mapped_result[result_key] = "개선대책을 생성할 수 없습니다."
        
        return mapped_result
        
    except Exception as e:
        st.error(f"JSON 파싱 중 오류 발생: {str(e)}")
        return None

# 데이터셋 옵션 (확장된 버전)
dataset_options = {
    "SWRO 건축공정 (건축)": "SWRO 건축공정 (건축)",
    "Civil (토목)": "Civil (토목)", 
    "Marine (토목)": "Marine (토목)",
    "SWRO 기계공사 (플랜트)": "SWRO 기계공사 (플랜트)",
    "SWRO 전기작업표준 (플랜트)": "SWRO 전기작업표준 (플랜트)",
    "샘플 데이터": "sample_data"
}

# 메인 애플리케이션 시작
# ----- 시스템 개요 탭 -----
with tabs[0]:
    st.markdown(f'<div class="sub-header">{texts["overview_header"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"""
        <div class="info-text">
        {texts["overview_text"]}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # AI 위험성평가 프로세스 다이어그램 (개선된 버전)
        st.markdown(f'<div class="data-card">', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center; margin-bottom: 15px; font-weight: bold; font-size: 1.1rem;">{texts["process_title"]}</div>', unsafe_allow_html=True)
        
        steps = texts["process_steps"]
        
        for i, step in enumerate(steps):
            phase_badge = '<span class="phase-badge">Phase 1</span>' if i < 4 else '<span class="phase-badge">Phase 2</span>'
            arrow = " ↓" if i < len(steps)-1 else ""
            st.markdown(f"**{i+1}. {step}** {phase_badge}{arrow}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 시스템 특징 (개선된 레이아웃)
    st.markdown(f'<div class="sub-header">{texts["features_title"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown(texts["phase1_features"], unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown(texts["phase2_features"], unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----- Phase 1: 위험성 평가 탭 (개선된 버전) -----
with tabs[1]:
    st.markdown(f'<div class="sub-header">{texts["phase1_header"]}</div>', unsafe_allow_html=True)
    
    # 설정 섹션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # API 키 입력
        api_key = st.text_input(texts["api_key_label"], type="password", key="api_key_phase1")
    
    with col2:
        # 데이터셋 선택
        selected_dataset_name = st.selectbox(
            texts["dataset_label"],
            options=list(dataset_options.keys()),
            key="dataset_selector_phase1"
        )
    
    # 데이터 로드 섹션 (개선된 UI)
    st.markdown("### " + texts['load_data_label'])
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button(texts["load_data_btn"], key="load_data_phase1", type="primary"):
            if not api_key:
                st.warning(texts["api_key_warning"])
            else:
                with st.spinner(texts["data_loading"]):
                    # 데이터 불러오기
                    df = load_data(dataset_options[selected_dataset_name])
                    
                    if df is not None:
                        # 데이터 정보 표시
                        st.session_state.current_dataset = selected_dataset_name
                        total_rows = len(df)
                        
                        # Train/Test 분할
                        if total_rows > 10:
                            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                        else:
                            train_df = df
                            test_df = df.sample(min(2, len(df)))
                        
                        # 리트리버 풀 구성
                        retriever_pool_df = train_df.copy()
                        retriever_pool_df['content'] = retriever_pool_df.apply(
                            lambda row: ' '.join([
                                str(row.get('작업활동 및 내용', '')),
                                str(row.get('유해위험요인 및 환경측면 영향', '')),
                                str(row.get('피해형태 및 환경영향', ''))
                            ]), axis=1
                        )
                        
                        texts_to_embed = retriever_pool_df['content'].tolist()
                        
                        # 임베딩 생성
                        with st.status("텍스트 임베딩 생성 중...", expanded=True) as status:
                            st.write(f"총 {len(texts_to_embed)}개 텍스트 처리 중...")
                            
                            openai.api_key = api_key
                            embeddings = embed_texts_with_openai(texts_to_embed, api_key=api_key)
                            
                            st.write("FAISS 인덱스 구성 중...")
                            # FAISS 인덱스 구성
                            embeddings_array = np.array(embeddings, dtype='float32')
                            dimension = embeddings_array.shape[1]
                            faiss_index = faiss.IndexFlatL2(dimension)
                            faiss_index.add(embeddings_array)
                            
                            status.update(label="인덱스 구성 완료!", state="complete")
                        
                        # 세션 상태에 저장
                        st.session_state.index = faiss_index
                        st.session_state.embeddings = embeddings_array
                        st.session_state.retriever_pool_df = retriever_pool_df
                        st.session_state.test_df = test_df
                        st.session_state.data_loaded = True
                        
                        # 성공 메시지
                        st.markdown(f"""
                        <div class="success-message">
                        ✅ {texts["data_load_success"].format(total_rows=total_rows)}
                        <br>📊 데이터셋: {selected_dataset_name}
                        <br>🔍 임베딩 차원: {dimension}
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.data_loaded:
            st.metric("데이터 상태", "✅ 로드됨", f"{len(st.session_state.retriever_pool_df)}개 항목")
    
    with col3:
        if st.session_state.data_loaded:
            st.metric("인덱스 상태", "✅ 구성됨", f"{st.session_state.embeddings.shape[1]}차원")
    
    # 유해위험요인 예측 섹션 (개선된 UI)
    st.markdown("### " + texts['hazard_prediction_header'])
    
    if st.session_state.index is None:
        st.markdown(f"""
        <div class="warning-message">
        ⚠️ {texts["load_first_warning"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.form("user_input_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_work = st.text_area(
                    texts["activity_label"], 
                    height=100,
                    placeholder="예: 굴착기를 이용한 토사 굴착 작업",
                    key="form_user_work"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    texts["predict_hazard_btn"], 
                    type="primary",
                    use_container_width=True
                )
            
        if submitted:
            if not user_work.strip():
                st.warning(texts["activity_warning"])
            else:
                with st.spinner(texts["predicting_hazard"]):
                    # 쿼리 임베딩
                    query_embedding = embed_texts_with_openai([user_work], api_key=api_key)[0]
                    query_embedding_array = np.array([query_embedding], dtype='float32')
                    
                    # 유사 문서 검색
                    k_similar = min(5, len(st.session_state.retriever_pool_df))
                    distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                    retrieved_docs = st.session_state.retriever_pool_df.iloc[indices[0]]
                    
                    # 신뢰도 점수 계산
                    similarity_scores = 1 / (1 + distances[0])  # 거리를 유사도로 변환
                    confidence = calculate_confidence_score(retrieved_docs, similarity_scores)
                    
                    # 결과 표시 섹션
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # 유사한 사례 표시 (개선된 UI)
                        st.markdown(f"#### {texts['similar_cases_header']}")
                        
                        for i, (_, doc) in enumerate(retrieved_docs.iterrows(), 1):
                            similarity_pct = similarity_scores[i-1] * 100
                            
                            st.markdown(f"""
                            <div class="similar-case">
                                <div style="display: flex; justify-content: between; align-items: center;">
                                    <strong>사례 {i}</strong>
                                    <span class="confidence-badge">유사도: {similarity_pct:.1f}%</span>
                                </div>
                                <strong>작업활동:</strong> {doc['작업활동 및 내용']}<br>
                                <strong>유해위험요인:</strong> {doc['유해위험요인 및 환경측면 영향']}<br>
                                <strong>위험도:</strong> 빈도 {doc['빈도']}, 강도 {doc['강도']}, T값 {doc['T']} (등급 {doc['등급']})
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # GPT 예측 결과
                        st.markdown(f"#### {texts['prediction_result_header']}")
                        
                        # 유해위험요인 예측
                        hazard_prompt = construct_prompt_phase1_hazard(
                            retrieved_docs, user_work, language=st.session_state.language
                        )
                        hazard_prediction = generate_with_gpt(
                            hazard_prompt, api_key=api_key, language=st.session_state.language
                        )
                        
                        # 빈도와 강도 예측
                        risk_prompt = construct_prompt_phase1_risk(
                            retrieved_docs, user_work, hazard_prediction, language=st.session_state.language
                        )
                        risk_prediction = generate_with_gpt(
                            risk_prompt, api_key=api_key, language=st.session_state.language
                        )
                        
                        # 결과 박스
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        
                        st.markdown(f"**{texts['activity_label']}** {user_work}")
                        st.markdown(f"**{texts['hazard_label']}** {hazard_prediction}")
                        
                        # 신뢰도 표시
                        st.markdown(f"""
                        <div style="text-align: right;">
                            <span class="confidence-badge">{texts.get('confidence_score', 'Confidence: {score}%').format(score=confidence)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 위험도 파싱 및 표시
                        parse_result = parse_gpt_output_phase1(risk_prediction, language=st.session_state.language)
                        if parse_result is not None:
                            f_val, i_val, t_val = parse_result
                            grade = determine_grade(t_val)
                            
                            # 메트릭으로 표시
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("빈도", f_val)
                                st.metric("T값", t_val)
                            with col_b:
                                st.metric("강도", i_val)
                                st.metric("위험등급", grade)
                            
                            # 위험등급에 따른 색상 표시
                            grade_colors = {'A': '#FF4444', 'B': '#FF8800', 'C': '#FFCC00', 'D': '#88CC00', 'E': '#44CC44'}
                            grade_color = grade_colors.get(grade, '#888888')
                            
                            st.markdown(f"""
                            <div style="background-color: {grade_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-top: 10px;">
                                위험등급: {grade}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 세션 상태에 결과 저장
                            assessment_data = {
                                'activity': user_work,
                                'hazard': hazard_prediction,
                                'frequency': f_val,
                                'intensity': i_val,
                                'T': t_val,
                                'grade': grade,
                                'confidence': confidence,
                                'dataset': st.session_state.current_dataset
                            }
                            st.session_state.last_assessment = assessment_data
                            
                            # 저장 버튼
                            if st.button("📊 " + texts.get("save_assessment", "Save Assessment"), key="save_phase1"):
                                save_assessment_to_history(assessment_data.copy())
                                st.success("✅ " + texts.get("assessment_saved", "Assessment saved!"))
                        
                        else:
                            st.error(texts["parsing_error"])
                            with st.expander("GPT 원문 응답 보기"):
                                st.write(risk_prediction)
                        
                        st.markdown('</div>', unsafe_allow_html=True)

# ----- Phase 2: 개선대책 생성 탭 (개선된 버전) -----
with tabs[2]:
    st.markdown(f'<div class="sub-header">{texts["phase2_header"]}</div>', unsafe_allow_html=True)
    
    # 설정 섹션
    col1, col2, col3 = st.columns(3)
    
    with col1:
        api_key_phase2 = st.text_input(texts["api_key_label"], type="password", key="api_key_phase2")
    
    with col2:
        target_language = st.selectbox(
            texts["language_select_label"],
            options=list(system_texts.keys()),
            index=list(system_texts.keys()).index(st.session_state.language),
            key="target_language"
        )
    
    with col3:
        input_method = st.radio(
            texts["input_method_label"],
            options=texts["input_methods"],
            index=0,
            key="input_method",
            horizontal=True
        )
    
    # 입력 데이터 처리
    if input_method == texts["input_methods"][0]:  # Phase 1 결과 사용
        if hasattr(st.session_state, 'last_assessment') and st.session_state.last_assessment:
            last_assessment = st.session_state.last_assessment
            
            # Phase 1 결과 표시
            st.markdown("### " + texts['phase1_results_header'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.markdown(f"**{texts['activity_label']}** {last_assessment['activity']}")
                st.markdown(f"**{texts['hazard_label']}** {last_assessment['hazard']}")
                st.markdown(f"**위험도:** 빈도 {last_assessment['frequency']}, 강도 {last_assessment['intensity']}, T값 {last_assessment['T']} (등급 {last_assessment['grade']})")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # 위험등급 시각화
                grade_colors = {'A': '#FF4444', 'B': '#FF8800', 'C': '#FFCC00', 'D': '#88CC00', 'E': '#44CC44'}
                grade_color = grade_colors.get(last_assessment['grade'], '#888888')
                
                st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <div style="font-size: 2rem; color: {grade_color}; font-weight: bold;">
                        {last_assessment['grade']}
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">위험등급</div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin-top: 10px;">
                        T = {last_assessment['T']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            activity_text = last_assessment['activity']
            hazard_text = last_assessment['hazard']
            frequency = last_assessment['frequency']
            intensity = last_assessment['intensity']
            T_value = last_assessment['T']
            
        else:
            st.markdown(f"""
            <div class="warning-message">
            ⚠️ {texts["phase1_first_warning"]}
            </div>
            """, unsafe_allow_html=True)
            activity_text = hazard_text = None
            frequency = intensity = T_value = None
    
    else:  # 직접 입력
        st.markdown("### 직접 입력")
        
        col1, col2 = st.columns(2)
        
        with col1:
            activity_text = st.text_area(texts["activity_label"], height=100, key="direct_activity")
            hazard_text = st.text_area(texts["hazard_label"], height=100, key="direct_hazard")
        
        with col2:
            frequency = st.slider(texts["frequency_label"], min_value=1, max_value=5, value=3, key="direct_freq")
            intensity = st.slider(texts["intensity_label"], min_value=1, max_value=5, value=3, key="direct_intensity")
            T_value = frequency * intensity
            
            st.markdown(f"""
            <div class="metric-container" style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold;">T = {T_value}</div>
                <div style="font-size: 0.9rem; color: #666;">등급: {determine_grade(T_value)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 개선대책 생성 섹션
    if activity_text and hazard_text and frequency and intensity and T_value:
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### " + texts.get("improvement_plan_header", "Improvement Measures"))
        
        with col2:
            generate_button = st.button(
                "🚀 " + texts["generate_improvement_btn"], 
                key="generate_improvement",
                type="primary",
                use_container_width=True
            )
        
        if generate_button:
            if not api_key_phase2:
                st.warning(texts["api_key_warning"])
            else:
                with st.spinner(texts["generating_improvement"]):
                    # 검색된 문서 준비
                    if st.session_state.retriever_pool_df is not None and st.session_state.index is not None:
                        # Phase 1에서 구성된 데이터 사용
                        retriever_pool_df = st.session_state.retriever_pool_df
                        
                        # 유사 문서 검색
                        query_text = f"{activity_text} {hazard_text}"
                        query_embedding = embed_texts_with_openai([query_text], api_key=api_key_phase2)[0]
                        query_embedding_array = np.array([query_embedding], dtype='float32')
                        
                        k_similar = min(3, len(retriever_pool_df))
                        distances, indices = st.session_state.index.search(query_embedding_array, k_similar)
                        retrieved_docs = retriever_pool_df.iloc[indices[0]]
                    else:
                        # 기본 데이터 사용
                        st.info(texts["no_data_warning"])
                        df = load_data("sample_data")
                        retrieved_docs = df.sample(min(3, len(df)))
                    
                    # 개선대책 생성 프롬프트
                    prompt = construct_prompt_phase2(
                        retrieved_docs, 
                        activity_text, 
                        hazard_text, 
                        frequency, 
                        intensity, 
                        T_value, 
                        target_language
                    )
                    
                    # GPT 호출
                    generated_output = generate_with_gpt(
                        prompt, 
                        api_key=api_key_phase2, 
                        language=target_language
                    )
                    
                    # 결과 파싱
                    parsed_result = parse_gpt_output_phase2(generated_output, language=target_language)
                    
                    if parsed_result:
                        # 결과 표시
                        improvement_plan = parsed_result.get("improvement", "")
                        improved_freq = parsed_result.get("improved_freq", 1)
                        improved_intensity = parsed_result.get("improved_intensity", 1)
                        improved_T = parsed_result.get("improved_t", improved_freq * improved_intensity)
                        rrr = parsed_result.get("reduction_rate", ((T_value - improved_T) / T_value * 100) if T_value > 0 else 0)
                        
                        # 결과 레이아웃
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # 개선대책
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown(f"#### 📋 {texts['improvement_plan_header']}")
                            st.markdown(improvement_plan.replace('1)', '\n1)').replace('2)', '\n2)').replace('3)', '\n3)').replace('4)', '\n4)'))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            # 위험도 개선 결과
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.markdown(f"#### 📊 {texts['risk_improvement_header']}")
                            
                            # 개선 전후 비교 차트
                            comparison_data = {
                                'Before': [frequency, intensity, T_value],
                                'After': [improved_freq, improved_intensity, improved_T]
                            }
                            comparison_df = pd.DataFrame(
                                comparison_data, 
                                index=['빈도', '강도', 'T값']
                            )
                            
                            st.bar_chart(comparison_df)
                            
                            # 위험 감소율
                            st.metric(
                                label="🎯 " + texts["risk_reduction_label"],
                                value=f"{rrr:.1f}%",
                                delta=f"-{T_value - improved_T}"
                            )
                            
                            # 등급 변화
                            before_grade = determine_grade(T_value)
                            after_grade = determine_grade(improved_T)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"**개선 전:** {before_grade}")
                            with col_b:
                                st.markdown(f"**개선 후:** {after_grade}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 전체 개선 결과 요약
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("#### 📈 개선 효과 요약")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("위험도(T)", improved_T, f"{improved_T - T_value}")
                        with col2:
                            st.metric("위험등급", after_grade, f"{before_grade}→{after_grade}")
                        with col3:
                            st.metric("감소율", f"{rrr:.1f}%")
                        with col4:
                            if rrr >= 70:
                                effectiveness = "매우 효과적"
                                color = "#4CAF50"
                            elif rrr >= 50:
                                effectiveness = "효과적"
                                color = "#FF9800"
                            else:
                                effectiveness = "보통"
                                color = "#f44336"
                            
                            st.markdown(f"""
                            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                                {effectiveness}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 저장 및 내보내기 버튼
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("💾 결과 저장", key="save_improvement"):
                                improvement_data = {
                                    'activity': activity_text,
                                    'hazard': hazard_text,
                                    'original_freq': frequency,
                                    'original_intensity': intensity,
                                    'original_T': T_value,
                                    'original_grade': before_grade,
                                    'improvement_plan': improvement_plan,
                                    'improved_freq': improved_freq,
                                    'improved_intensity': improved_intensity,
                                    'improved_T': improved_T,
                                    'improved_grade': after_grade,
                                    'reduction_rate': rrr,
                                    'language': target_language
                                }
                                save_assessment_to_history(improvement_data)
                                st.success("✅ 개선대책이 저장되었습니다!")
                        
                        with col2:
                            # Excel 내보내기 버튼
                            st.download_button(
                                label="📄 Excel 다운로드",
                                data=pd.DataFrame([improvement_data]).to_csv(index=False),
                                file_name=f"improvement_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col3:
                            # 새 평가 버튼
                            if st.button("🔄 새 평가", key="new_assessment"):
                                st.session_state.last_assessment = None
                                st.experimental_rerun()
                    
                    else:
                        st.error(texts["parsing_error_improvement"])
                        with st.expander("GPT 원문 응답 보기"):
                            st.write(generated_output)

# ----- 평가 이력 탭 -----
with tabs[3]:
    st.markdown(f'<div class="sub-header">{texts.get("history_header", "Assessment History")}</div>', unsafe_allow_html=True)
    
    if st.session_state.assessment_history:
        # 이력 통계
        col1, col2, col3, col4 = st.columns(4)
        
        history_df = pd.DataFrame(st.session_state.assessment_history)
        
        with col1:
            st.metric(
                texts.get("total_assessments", "Total Assessments"), 
                len(st.session_state.assessment_history)
            )
        
        with col2:
            if 'grade' in history_df.columns:
                high_risk_count = len(history_df[history_df['grade'] == 'A'])
                st.metric(
                    texts.get("high_risk_count", "High Risk (A Grade)"), 
                    high_risk_count
                )
        
        with col3:
            if 'T' in history_df.columns:
                avg_risk = history_df['T'].mean()
                st.metric(
                    texts.get("avg_risk_score", "Average Risk Score"), 
                    f"{avg_risk:.1f}"
                )
        
        with col4:
            if 'reduction_rate' in history_df.columns:
                avg_improvement = history_df['reduction_rate'].mean()
                st.metric(
                    texts.get("improvement_rate", "Improvement Rate"), 
                    f"{avg_improvement:.1f}%"
                )
        
        # 이력 테이블
        st.markdown("### 평가 이력 상세")
        
        # 데이터 정리
        display_columns = ['timestamp', 'activity', 'hazard', 'T', 'grade']
        if 'reduction_rate' in history_df.columns:
            display_columns.append('reduction_rate')
        
        display_df = history_df[display_columns].copy()
        display_df.columns = ['시간', '작업활동', '유해위험요인', 'T값', '등급', '개선율(%)'][:len(display_columns)]
        
        # 테이블 표시 (페이지네이션)
        page_size = 10
        total_pages = (len(display_df) - 1) // page_size + 1
        
        if total_pages > 1:
            page = st.selectbox("페이지 선택", range(1, total_pages + 1))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            display_df = display_df.iloc[start_idx:end_idx]
        
        st.dataframe(display_df, use_container_width=True)
        
        # 이력 다운로드
        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                label="📊 이력 다운로드",
                data=pd.DataFrame(st.session_state.assessment_history).to_csv(index=False),
                file_name=f"assessment_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>아직 저장된 평가 이력이 없습니다</h3>
            <p>Phase 1 또는 Phase 2에서 평가를 수행하고 저장해보세요.</p>
        </div>
        """, unsafe_allow_html=True)

# ----- 통계 분석 탭 -----
with tabs[4]:
    st.markdown(f'<div class="sub-header">{texts.get("statistics_header", "Statistical Analysis")}</div>', unsafe_allow_html=True)
    
    if st.session_state.assessment_history:
        # 시각화 생성
        charts = create_risk_visualization(st.session_state.assessment_history)
        
        if charts:
            fig_grade, fig_trend = charts
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 위험등급 분포")
                st.plotly_chart(fig_grade, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 월별 평가 추이")
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # 상세 분석
        history_df = pd.DataFrame(st.session_state.assessment_history)
        
        st.markdown("### 📋 데이터 인사이트")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("#### 위험도 분석")
            
            if 'T' in history_df.columns:
                # T값 분포
                t_values = history_df['T']
                st.markdown(f"- **평균 T값:** {t_values.mean():.2f}")
                st.markdown(f"- **최고 T값:** {t_values.max()}")
                st.markdown(f"- **최저 T값:** {t_values.min()}")
                st.markdown(f"- **표준편차:** {t_values.std():.2f}")
                
                # T값 히스토그램
                fig_hist = px.histogram(
                    history_df, x='T', 
                    title="T값 분포",
                    nbins=10
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("#### 개선 효과 분석")
            
            if 'reduction_rate' in history_df.columns:
                improvement_data = history_df.dropna(subset=['reduction_rate'])
                
                if len(improvement_data) > 0:
                    reduction_rates = improvement_data['reduction_rate']
                    st.markdown(f"- **평균 개선율:** {reduction_rates.mean():.1f}%")
                    st.markdown(f"- **최고 개선율:** {reduction_rates.max():.1f}%")
                    st.markdown(f"- **최저 개선율:** {reduction_rates.min():.1f}%")
                    
                    # 개선율 분포
                    fig_improvement = px.box(
                        improvement_data, y='reduction_rate',
                        title="개선율 분포"
                    )
                    st.plotly_chart(fig_improvement, use_container_width=True)
                else:
                    st.markdown("개선대책 데이터가 없습니다.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 작업유형별 분석
        if 'activity' in history_df.columns:
            st.markdown("### 🏗️ 작업유형별 위험도 분석")
            
            # 간단한 키워드 분류
            def classify_work_type(activity):
                activity_lower = str(activity).lower()
                if any(word in activity_lower for word in ['굴착', 'excavation', 'dig']):
                    return '굴착작업'
                elif any(word in activity_lower for word in ['용접', 'welding', 'weld']):
                    return '용접작업'
                elif any(word in activity_lower for word in ['운반', 'transport', 'carry']):
                    return '운반작업'
                elif any(word in activity_lower for word in ['설치', 'install', 'assembly']):
                    return '설치작업'
                elif any(word in activity_lower for word in ['해체', 'demolition', 'dismantle']):
                    return '해체작업'
                else:
                    return '기타작업'
            
            history_df['work_type'] = history_df['activity'].apply(classify_work_type)
            
            # 작업유형별 통계
            work_type_stats = history_df.groupby('work_type').agg({
                'T': ['mean', 'max', 'count'],
                'grade': lambda x: (x == 'A').sum()
            }).round(2)
            
            work_type_stats.columns = ['평균_T값', '최대_T값', '평가_건수', 'A등급_건수']
            
            st.dataframe(work_type_stats, use_container_width=True)
            
            # 작업유형별 위험도 차트
            if len(work_type_stats) > 1:
                fig_worktype = px.bar(
                    x=work_type_stats.index,
                    y=work_type_stats['평균_T값'],
                    title="작업유형별 평균 위험도",
                    labels={'x': '작업유형', 'y': '평균 T값'}
                )
                st.plotly_chart(fig_worktype, use_container_width=True)
        
        # 데이터 내보내기
        st.markdown("### 📤 데이터 내보내기")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 전체 이력 다운로드
            st.download_button(
                label="📊 전체 이력 CSV",
                data=history_df.to_csv(index=False),
                file_name=f"full_assessment_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # 고위험 데이터만 다운로드
            if 'grade' in history_df.columns:
                high_risk_df = history_df[history_df['grade'].isin(['A', 'B'])]
                if len(high_risk_df) > 0:
                    st.download_button(
                        label="⚠️ 고위험 데이터 CSV",
                        data=high_risk_df.to_csv(index=False),
                        file_name=f"high_risk_assessments_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            # 개선대책 데이터만 다운로드
            if 'improvement_plan' in history_df.columns:
                improvement_df = history_df.dropna(subset=['improvement_plan'])
                if len(improvement_df) > 0:
                    st.download_button(
                        label="💡 개선대책 CSV",
                        data=improvement_df.to_csv(index=False),
                        file_name=f"improvement_plans_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>통계 분석을 위한 데이터가 없습니다</h3>
            <p>몇 건의 평가를 수행한 후 이 탭에서 통계를 확인할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)

# ----- 푸터 섹션 -----
st.markdown('<hr style="margin-top: 50px; border: 1px solid #e0e0e0;">', unsafe_allow_html=True)

# 시스템 정보 및 로고
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h4 style="color: #1976D2;">🏗️ 건설 안전 AI</h4>
        <p style="color: #666; font-size: 0.9rem;">
            LLM 기반 위험성평가<br>
            자동화 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # 로고 표시 (실제 파일이 있는 경우)
    if os.path.exists("cau.png"):
        cau_logo = Image.open("cau.png")
        st.image(cau_logo, width=120)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="background: #f5f5f5; border-radius: 10px; padding: 20px; margin: 10px;">
                <strong>중앙대학교</strong><br>
                <small>건설환경플랜트공학과</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if os.path.exists("doosan.png"):
        doosan_logo = Image.open("doosan.png")
        st.image(doosan_logo, width=150)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="background: #f5f5f5; border-radius: 10px; padding: 20px; margin: 10px;">
                <strong>두산에너빌리티</strong><br>
                <small>EHS 디지털혁신</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# 버전 정보
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.8rem; margin-top: 20px;">
    AI Risk Assessment System v2.0 | 개선된 버전 | Last Updated: 2025-05-30
</div>
""", unsafe_allow_html=True)

# 개발자 노트 (선택적 표시)
with st.expander("🔧 개발자 노트 및 개선사항"):
    st.markdown("""
    ### 주요 개선사항
    
    ✅ **UI/UX 개선**
    - 현대적이고 직관적인 인터페이스 디자인
    - 그라데이션과 그림자 효과로 시각적 품질 향상
    - 반응형 레이아웃으로 다양한 화면 크기 지원
    
    ✅ **기능 확장**
    - 평가 이력 관리 및 통계 분석 탭 추가
    - 신뢰도 점수 표시로 AI 예측 품질 가시화
    - 다양한 데이터 내보내기 옵션 제공
    
    ✅ **성능 최적화**
    - 배치 처리로 임베딩 생성 효율성 개선
    - 더 정확한 JSON 파싱 로직
    - 오류 처리 및 사용자 피드백 강화
    
    ✅ **데이터 처리 개선**
    - 더 포괄적인 Excel 파일 처리
    - 자동 컬럼명 매핑 및 데이터 정규화
    - 누락 데이터에 대한 견고한 처리
    
    ✅ **시각화 강화**
    - Plotly를 활용한 인터랙티브 차트
    - 위험등급별 색상 코딩
    - 실시간 메트릭 표시
    
    ### 기술적 특징
    - **멀티 언어 지원**: 한국어, 영어, 중국어
    - **실시간 AI 분석**: OpenAI GPT-4 기반
    - **의미론적 검색**: FAISS 벡터 인덱싱
    - **확장 가능한 아키텍처**: 모듈화된 함수 구조
    """)

# 디버그 정보 (개발 모드에서만 표시)
if st.sidebar.checkbox("🐛 디버그 모드", key="debug_mode"):
    st.sidebar.markdown("### 세션 상태")
    st.sidebar.json({
        "데이터 로드됨": st.session_state.data_loaded,
        "현재 데이터셋": st.session_state.current_dataset,
        "언어": st.session_state.language,
        "평가 이력 수": len(st.session_state.assessment_history),
        "마지막 평가": bool(st.session_state.last_assessment)
    })
