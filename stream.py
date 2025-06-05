import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ----------------- 시스템 다국어 텍스트 -----------------
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_phase1": "위험성 평가 (Phase 1)",
        "tab_phase2": "개선대책 생성 (Phase 2)",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": "두산에너빌리티 AI Risk Assessment는 국내 및 해외 건설현장 '수시위험성평가' 및 '노동부 중대재해 사례'를 학습하여 개발된 자동 위험성평가 프로그램입니다. 생성된 위험성평가는 반드시 수시 위험성평가 심의회를 통해 검증 후 사용하시기 바랍니다.",
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
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "predict_hazard_btn": "유해위험요인 예측하기",
        "activity_warning": "작업활동을 입력하세요.",
        "similar_cases_header": "유사한 사례",
        "result_table_columns": [
            "작업활동 및 내용 Work Sequence",
            "유해위험요인 및 환경측면 영향 Hazardous Factors",
            "EHS",
            "빈도 likelihood",
            "강도 severity",
            "개선대책 및 세부관리방안 Control Measures",
            "개선담당자 In Charge",
            "개선일자 Correction Due Date"
        ],
        "parsing_error": "위험성 평가 결과를 파싱할 수 없습니다.",
        "phase2_header": "개선대책 자동 생성 (Phase 2)",
        "language_select_label": "개선대책 언어 선택:",
        "input_method_label": "입력 방식 선택:",
        "input_methods": ["Phase 1 평가 결과 사용", "직접 입력"],
        "phase1_results_header": "Phase 1 평가 결과",
        "phase1_first_warning": "먼저 Phase 1에서 위험성 평가를 수행하세요.",
        "hazard_label": "유해위험요인:",
        "frequency_label": "빈도 (1-5):",
        "intensity_label": "강도 (1-5):",
        "generate_improvement_btn": "개선대책 생성",
        "generating_improvement": "개선대책을 생성하는 중...",
        "no_data_warning": "Phase 1에서 데이터 로드 및 인덱스 구성을 완료하지 않았습니다. 기본 예시를 사용합니다.",
        "improvement_plan_header": "개선대책",
        "risk_improvement_header": "개선 후 위험성",
        "excel_export": "📥 결과 Excel 다운로드",
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_phase1": "Risk Assessment (Phase 1)",
        "tab_phase2": "Improvement Measures (Phase 2)",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": "Doosan Enerbility AI Risk Assessment is an automated program trained on both on-demand risk-assessment reports from domestic and overseas construction sites and major-accident cases compiled by Korea's Ministry of Employment and Labor. Please ensure that every generated assessment is reviewed and approved by the On-Demand Risk Assessment Committee before it is used.",
        "features_title": "System Features and Components",
        "phase1_features": """
        #### Phase 1: Risk Assessment Automation
        - Learning risk assessment data according to work activities by process
        - Automatic hazard prediction when work activities are entered
        - Similar case search and display
        - Risk level (frequency, intensity, T) measurement based on large language models (LLM)
        - Automatic analysis of Excel-based process-specific risk assessment data
        - Automatic risk grade (A-E) calculation
        """,
        "phase2_features": """
        #### Phase 2: Automatic Generation of Improvement Measures
        - Automatic generation of customized improvement measures for risk factors
        - Multilingual (Korean/English/Chinese) improvement measure generation support
        - Automatic comparative analysis of risk level (T) before and after improvement
        - Quantitative calculation of Risk Reduction Rate (RRR)
        - Building a database of optimal improvement measures by work type/process
        """,
        "phase1_header": "Risk Assessment Automation (Phase 1)",
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_btn": "Load Data and Configure Index",
        "api_key_warning": "Please enter an OpenAI API key to continue.",
        "data_loading": "Loading data and configuring index...",
        "demo_limit_info": "For demo purposes, only embedding {max_texts} texts. In a real environment, all data should be processed.",
        "data_load_success": "Data load and index configuration complete! (Total {max_texts} items processed)",
        "load_first_warning": "Please click the [Load Data and Configure Index] button first.",
        "activity_label": "Work Activity:",
        "predict_hazard_btn": "Predict Hazards",
        "activity_warning": "Please enter a work activity.",
        "similar_cases_header": "Similar Cases",
        "result_table_columns": [
            "Work Sequence",
            "Hazardous Factors",
            "EHS",
            "likelihood",
            "severity",
            "Control Measures",
            "In Charge",
            "Correction Due Date"
        ],
        "parsing_error": "Unable to parse risk assessment results.",
        "phase2_header": "Automatic Generation of Improvement Measures (Phase 2)",
        "language_select_label": "Select Language for Improvement Measures:",
        "input_method_label": "Select Input Method:",
        "input_methods": ["Use Phase 1 Assessment Results", "Direct Input"],
        "phase1_results_header": "Phase 1 Assessment Results",
        "phase1_first_warning": "Please perform a risk assessment in Phase 1 first.",
        "hazard_label": "Hazard:",
        "frequency_label": "Frequency (1-5):",
        "intensity_label": "Intensity (1-5):",
        "generate_improvement_btn": "Generate Improvement Measures",
        "generating_improvement": "Generating improvement measures...",
        "no_data_warning": "Data loading and index configuration was not completed in Phase 1. Using basic examples.",
        "improvement_plan_header": "Control Measures",
        "risk_improvement_header": "Post-Improvement Risk",
        "excel_export": "📥 Download Excel Results",
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_phase1": "风险评估 (第1阶段)",
        "tab_phase2": "改进措施 (第2阶段)",
        "overview_header": "基于LLM的风险评估系统",
        "overview_text": "Doosan Enerbility AI 风险评估系统是一款自动化风险评估程序，基于国内外施工现场的'临时风险评估'数据以及韩国雇佣劳动部的重大事故案例进行训练开发而成。生成的风险评估结果必须经过临时风险评估审议委员会的审核后方可使用。",
        "features_title": "系统特点和组件",
        "phase1_features": """
        #### 第1阶段：风险评估自动化
        - 按工序学习与工作活动相关的风险评估数据
        - 输入工作活动时自动预测危害因素
        - 相似案例搜索和显示
        - 基于大型语言模型(LLM)的风险等级（频率、强度、T值）测量
        - 自动分析基于Excel的特定工序风险评估数据
        - 自动计算风险等级(A-E)
        """,
        "phase2_features": """
        #### 第2阶段：自动生成改进措施
        - 为风险因素自动生成定制的改进措施
        - 多语言（韩语/英语/中文）改进措施生成支持
        - 改进前后风险等级（T值）的自动比较分析
        - 风险降低率(RRR)的定量计算
        - 建立按工作类型/工序的最佳改进措施数据库
        """,
        "phase1_header": "风险评估自动化 (第1阶段)",
        "api_key_label": "输入OpenAI API密钥：",
        "dataset_label": "选择数据集",
        "load_data_btn": "加载数据和配置索引",
        "api_key_warning": "请输入OpenAI API密钥以继续。",
        "data_loading": "正在加载数据和配置索引...",
        "demo_limit_info": "出于演示目的，仅嵌入{max_texts}个文本。在实际环境中，应处理所有数据。",
        "data_load_success": "数据加载和索引配置完成！（共处理{max_texts}个项目）",
        "load_first_warning": "请先点击[加载数据和配置索引]按钮。",
        "activity_label": "工作活动：",
        "predict_hazard_btn": "预测危害",
        "activity_warning": "请输入工作活动。",
        "similar_cases_header": "相似案例",
        "result_table_columns": [
            "工作活动 Work Sequence",
            "危害因素 Hazardous Factors",
            "EHS",
            "频率 likelihood",
            "强度 severity",
            "控制措施 Control Measures",
            "责任人 In Charge",
            "整改日期 Correction Due Date"
        ],
        "parsing_error": "无法解析风险评估结果。",
        "phase2_header": "自动生成改进措施 (第2阶段)",
        "language_select_label": "选择改进措施的语言：",
        "input_method_label": "选择输入方法：",
        "input_methods": ["使用第1阶段评估结果", "直接输入"],
        "phase1_results_header": "第1阶段评估结果",
        "phase1_first_warning": "请先在第1阶段进行风险评估。",
        "hazard_label": "危害：",
        "frequency_label": "频率 (1-5)：",
        "intensity_label": "强度 (1-5)：",
        "generate_improvement_btn": "生成改进措施",
        "generating_improvement": "正在生成改进措施...",
        "no_data_warning": "在第1阶段未完成数据加载和索引配置。使用基本示例。",
        "improvement_plan_header": "控制措施 Control Measures",
        "risk_improvement_header": "整改后风险 Post-Improvement Risk",
        "excel_export": "📥 下载Excel结果",
    }
}

# ----------------- 페이지 스타일 -----------------
st.set_page_config(page_title="Artificial Intelligence Risk Assessment", page_icon="🛠️", layout="wide")
st.markdown(
    """
    <style>
    .main-header{font-size:2.5rem;color:#1E88E5;text-align:center;margin-bottom:1rem}
    .sub-header{font-size:1.8rem;color:#0D47A1;margin-top:2rem;margin-bottom:1rem}
    .metric-container{background-color:#f0f2f6;border-radius:10px;padding:20px;box-shadow:2px 2px 5px rgba(0,0,0,0.1)}
    .result-box{background-color:#f8f9fa;border-radius:10px;padding:15px;margin-top:10px;margin-bottom:10px;border-left:5px solid #4CAF50}
    .phase-badge{background-color:#4CAF50;color:white;padding:5px 10px;border-radius:15px;font-size:0.8rem;margin-right:10px}
    .similar-case{background-color:#f1f8e9;border-radius:8px;padding:12px;margin-bottom:8px;border-left:4px solid #689f38}
    .language-selector{position:absolute;top:10px;right:10px;z-index:1000}
    .calculation-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin:20px 0}
    .risk-grade-a{background-color:#ff1744;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
    .risk-grade-b{background-color:#ff9800;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
    .risk-grade-c{background-color:#ffc107;color:black;padding:5px 10px;border-radius:5px;font-weight:bold}
    .risk-grade-d{background-color:#4caf50;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
    .risk-grade-e{background-color:#2196f3;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- 세션 상태 초기화 -----------------
ss = st.session_state
for key, default in {
    "language": "Korean",
    "index": None,
    "embeddings": None,
    "retriever_pool_df": None,
    "last_assessment": None
}.items():
    if key not in ss:
        ss[key] = default

# ----------------- 언어 선택 -----------------
col0, colLang = st.columns([6, 1])
with colLang:
    lang = st.selectbox(
        "언어 선택",
        list(system_texts.keys()),
        index=list(system_texts.keys()).index(ss.language),
        label_visibility="hidden"
    )
    ss.language = lang
texts = system_texts[ss.language]

# ----------------- 헤더 -----------------
st.markdown(f'<div class="main-header">{texts["title"]}</div>', unsafe_allow_html=True)

# ----------------- 탭 구성 -----------------
tabs = st.tabs([texts["tab_overview"], f"{texts['tab_phase1']} & {texts['tab_phase2']}"])

# -----------------------------------------------------------------------------  
# --------------------------- 유틸리티 함수 -------------------------------------  
# -----------------------------------------------------------------------------  

def determine_grade(value: int):
    """위험도 등급 분류"""
    if 16 <= value <= 25:
        return 'A'
    if 10 <= value <= 15:
        return 'B'
    if 5 <= value <= 9:
        return 'C'
    if 3 <= value <= 4:
        return 'D'
    if 1 <= value <= 2:
        return 'E'
    return 'Unknown' if ss.language != 'Korean' else '알 수 없음'

def get_grade_color(grade):
    """위험등급별 색상 반환"""
    colors = {
        'A': '#ff1744',
        'B': '#ff9800',
        'C': '#ffc107',
        'D': '#4caf50',
        'E': '#2196f3',
    }
    return colors.get(grade, '#808080')

def compute_rrr(original_t, improved_t):
    """위험 감소율 계산"""
    if original_t == 0:
        return 0.0
    return ((original_t - improved_t) / original_t) * 100

def _extract_improvement_info(row):
    """
    유사 사례 한 건에서 - 개선대책 / 개선 후 빈도·강도·T 값을 추출
    """
    plan_cols = [c for c in row.index if re.search(r'개선대책|Improvement|改进', c, re.I)]
    plan = row[plan_cols[0]] if plan_cols else ""

    cand_sets = [
        ('개선 후 빈도', '개선 후 강도', '개선 후 T'),
        ('개선빈도', '개선강도', '개선T'),
        ('improved_frequency', 'improved_intensity', 'improved_T'),
        ('改进后频率', '改进后强度', '改进后T값'),
    ]
    imp_f, imp_i, imp_t = None, None, None
    for f, i, t in cand_sets:
        if f in row and i in row and t in row:
            imp_f, imp_i, imp_t = int(row[f]), int(row[i]), int(row[t])
            break

    if imp_f is None:
        orig_f, orig_i = int(row['빈도']), int(row['강도'])
        imp_f = max(1, orig_f - 1)
        imp_i = max(1, orig_i - 1)
        imp_t = imp_f * imp_i

    return plan, imp_f, imp_i, imp_t

@st.cache_data(show_spinner=False)
def load_data(selected_dataset_name: str):
    """데이터 로드 및 전처리"""
    try:
        if os.path.exists(f"{selected_dataset_name}.xlsx"):
            df = pd.read_excel(f"{selected_dataset_name}.xlsx")
        else:
            return create_sample_data()

        if "삭제 Del" in df.columns:
            df.drop(["삭제 Del"], axis=1, inplace=True)

        df = df.dropna(how='all')

        column_mapping = {
            "작업활동 및 내용\nWork & Contents": "작업활동 및 내용",
            "유해위험요인 및 환경측면 영향\nHazard & Risk": "유해위험요인 및 환경측면 영향",
            "피해형태 및 환경영향\nDamage & Effect": "피해형태 및 환경영향",
            "개선대책 및 세부관리방안\nCorrective Action": "개선대책"
        }
        df.rename(columns=column_mapping, inplace=True)

        numeric_columns = ['빈도', '강도']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '빈도' not in df.columns:
            df['빈도'] = 3
        if '강도' not in df.columns():
            df['강도'] = 3

        df["T"] = df["빈도"] * df["강도"]
        df["등급"] = df["T"].apply(determine_grade)

        if "개선대책" not in df.columns:
            alt_cols = [c for c in df.columns if "개선" in c or "대책" in c or "Corrective" in c]
            if alt_cols:
                df.rename(columns={alt_cols[0]: "개선대책"}, inplace=True)
            else:
                df["개선대책"] = "안전 교육 실시 및 보호구 착용"

        required_cols = [
            "작업활동 및 내용",
            "유해위험요인 및 환경측면 영향",
            "피해형태 및 환경영향",
            "빈도",
            "강도",
            "T",
            "등급",
            "개선대책"
        ]
        final_cols = [col for col in required_cols if col in df.columns]
        df = df[final_cols]

        df = df.fillna({
            "작업활동 및 내용": "일반 작업",
            "유해위험요인 및 환경측면 영향": "일반적 위험",
            "피해형태 및 환경영향": "부상",
            "개선대책": "안전 조치 수행"
        })

        return df

    except Exception as e:
        st.warning(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        st.info("샘플 데이터를 사용합니다.")
        return create_sample_data()

def create_sample_data():
    """샘플 데이터 생성"""
    data = {
        "작업활동 및 내용": [
            "임시 현장 저장소에서 포크리프트를 이용한 철골 구조재 하역작업",
            "콘크리트/CMU 블록 설치 작업",
            "굴착 및 되메우기 작업",
            "고소 작업대를 이용한 외벽 작업",
            "용접 작업"
        ],
        "유해위험요인 및 환경측면 영향": [
            "다중 인양으로 인한 적재물 낙하",
            "불충분한 작업 발판으로 인한 추락",
            "굴착벽 붕괴로 인한 매몰",
            "안전대 미착용으로 인한 추락",
            "용접 흄 및 화재 위험"
        ],
        "피해형태 및 환경영향": [
            "타박상",
            "골절",
            "매몰",
            "추락사",
            "화상"
        ],
        "빈도": [3, 3, 2, 4, 2],
        "강도": [5, 4, 5, 5, 3],
        "개선대책": [
            "1) 다수의 철골재를 함께 인양하지 않도록 관리 2) 치수, 중량, 형상이 다른 재료를 함께 인양하지 않도록 관리",
            "1) 비계대 누락된 목판 설치 2) 안전대 부착설비 설치 및 사용 3) 비계 변경 시 타공종 외 작업자 작업 금지",
            "1) 적절한 사면 기울기 유지 2) 굴착면 보강 3) 정기적 지반 상태 점검",
            "1) 안전대 착용 의무화 2) 작업 전 안전교육 실시 3) 추락방지망 설치",
            "1) 적절한 환기시설 설치 2) 화재 예방 조치 3) 보호구 착용"
        ]
    }
    df = pd.DataFrame(data)
    df["T"] = df["빈도"] * df["강도"]
    df["등급"] = df["T"].apply(determine_grade)
    return df

def embed_texts_with_openai(texts, api_key, model="text-embedding-3-large"):
    """
    OpenAI 공식 API를 이용한 텍스트 임베딩 생성
    (오류 발생 시 예외를 화면에 출력하도록 수정)
    """
    if not api_key:
        st.error("API 키가 설정되어 있지 않습니다.")
        return []

    client = OpenAI(api_key=api_key)
    embeddings = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        processed_texts = [str(txt).replace("\n", " ").strip() for txt in batch_texts]

        try:
            resp = client.embeddings.create(
                model=model,
                input=processed_texts
            )
            for item in resp.data:
                embeddings.append(item.embedding)
        except Exception as e:
            st.error(f"임베딩 호출 실패 (배치 {i}): {e}")
            for _ in batch_texts:
                embeddings.append([0.0] * 1536)

    return embeddings

def generate_with_gpt(prompt, api_key, language, model="gpt-4o", max_retries=3):
    """OpenAI 공식 API를 이용한 GPT 생성 함수"""
    if not api_key:
        st.error("API 키가 설정되어 있지 않습니다.")
        return ""

    client = OpenAI(api_key=api_key)
    sys_prompts = {
        "Korean": "You are a construction site risk assessment expert. Provide accurate, practical responses in Korean.",
        "English": "You are a construction site risk assessment expert. Provide accurate, practical responses in English.",
        "Chinese": "您是建筑工地风险评估专家。请用中文提供准确实用的回答。"
    }

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompts.get(language, sys_prompts["English"])},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"GPT 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                return ""
            else:
                st.warning(f"GPT 호출 재시도 중... ({attempt + 1}/{max_retries})")
                continue

def construct_prompt_phase1_hazard(retrieved_docs, activity_text, language="English"):
    """유해위험요인 예측 프롬프트 (영문 내부 처리)"""
    prompt_templates = {
        "English": {
            "intro": "Here are examples of work activities and associated hazards at construction sites:\n\n",
            "example_format": "Case {i}:\n- Work Activity: {activity}\n- Hazardous Factors: {hazard}\n\n",
            "query_format": "Based on the above cases, please predict the main hazardous factors for the following work activity:\n\nWork Activity: {activity}\n\nPredicted Hazardous Factors: "
        }
    }

    template = prompt_templates["English"]
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            activity = doc["작업활동 및 내용"]
            hazard = doc["유해위험요인 및 환경측면 영향"]
            if pd.notna(activity) and pd.notna(hazard):
                retrieved_examples.append((activity, hazard))
        except:
            continue

    prompt = template["intro"]
    for i, (act, haz) in enumerate(retrieved_examples[:5], 1):
        prompt += template["example_format"].format(i=i, activity=act, hazard=haz)

    prompt += template["query_format"].format(activity=activity_text)
    return prompt

def construct_prompt_phase1_risk(retrieved_docs, activity_text, hazard_text, language="English"):
    """위험도 평가 프롬프트 (영문 내부 처리)"""
    prompt = (
        "Construction site risk assessment criteria:\n"
        "- Frequency (1-5): 1=Very Rare, 2=Rare, 3=Occasional, 4=Frequent, 5=Very Frequent\n"
        "- Severity (1-5): 1=Minor Injury, 2=Light Injury, 3=Moderate Injury, 4=Serious Injury, 5=Fatality\n"
        "- T-value = Frequency × Severity\n\n"
        "Reference cases:\n\n"
    )

    for i, row in enumerate(retrieved_docs.head(3).itertuples(), 1):
        inp = f"{row._2} - {row._3}"  # 작업활동 및 유해위험요인
        freq = int(row._4)            # 빈도
        sev = int(row._5)             # 강도
        t_val = freq * sev
        prompt += f"Case {i}:\nInput: {inp}\nAssessment: Frequency={freq}, Severity={sev}, T-value={t_val}\n\n"

    prompt += (
        f"Based on the above criteria and cases, please assess the following:\n\n"
        f"Work Activity: {activity_text}\n"
        f"Hazardous Factors: {hazard_text}\n\n"
        f"Respond in JSON format: " 
        f'{{"frequency": number, "severity": number, "T": number}}'
    )
    return prompt

def parse_gpt_output_phase1(gpt_output, language="English"):
    """GPT 출력 파싱 (Phase 1)"""
    pattern = r'\{"frequency":\s*([1-5]),\s*"severity":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    match = re.search(pattern, gpt_output)
    if match:
        freq = int(match.group(1))
        sev = int(match.group(2))
        t_val = int(match.group(3))
        return freq, sev, t_val

    fallback = re.findall(r'\b([1-5])\b', gpt_output)
    if len(fallback) >= 2:
        f = int(fallback[0])
        s = int(fallback[1])
        return f, s, f * s

    return None

def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, severity, T, language="English"):
    """개선대책 생성 프롬프트 (영문 내부 처리)"""
    examples = []
    for _, row in retrieved_docs.head(3).iterrows():
        plan_candidates = [c for c in row.index if "개선대책" in c or "Improvement" in c or "改进" in c]
        plan = row[plan_candidates[0]] if plan_candidates and pd.notna(row[plan_candidates[0]]) else ""
        orig_f = int(row["빈도"])
        orig_s = int(row["강도"])
        orig_t = orig_f * orig_s
        new_f = max(1, orig_f - 1)
        new_s = max(1, orig_s - 1)
        new_t = new_f * new_s
        examples.append({
            "activity": row["작업활동 및 내용"],
            "hazard": row["유해위험요인 및 환경측면 영향"],
            "orig_f": orig_f,
            "orig_s": orig_s,
            "orig_t": orig_t,
            "plan": plan,
            "new_f": new_f,
            "new_s": new_s,
            "new_t": new_t
        })

    prompt = ""
    for i, ex in enumerate(examples, 1):
        prompt += (
            f"Example {i}:\n"
            f"Input Work Activity: {ex['activity']}\n"
            f"Input Hazardous Factors: {ex['hazard']}\n"
            f"Input Original Frequency: {ex['orig_f']}\n"
            f"Input Original Severity: {ex['orig_s']}\n"
            f"Input Original T-value: {ex['orig_t']}\n"
            f"Output (JSON):\n"
            "{\n"
            f'  "control_measures": "{ex["plan"]}",\n'
            f'  "post_frequency": {ex["new_f"]},\n'
            f'  "post_severity": {ex["new_s"]},\n'
            f'  "post_T": {ex["new_t"]},\n'
            f'  "reduction_rate": {compute_rrr(ex["orig_t"], ex["new_t"]):.2f}\n'
            "}\n\n"
        )

    prompt += (
        f"Now please provide improvement measures in JSON format for the following:\n\n"
        f"Work Activity: {activity_text}\n"
        f"Hazardous Factors: {hazard_text}\n"
        f"Original Frequency: {freq}\n"
        f"Original Severity: {severity}\n"
        f"Original T-value: {T}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "control_measures": "string",\n'
        '  "post_frequency": number,\n'
        '  "post_severity": number,\n'
        '  "post_T": number,\n'
        '  "reduction_rate": number\n'
        "}"
    )
    return prompt

def parse_gpt_output_phase2(gpt_output, language="English"):
    """Phase 2 출력 파싱"""
    json_match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
    if not json_match:
        return None

    import json
    try:
        data = json.loads(json_match.group())
        return {
            "control_measures": data.get("control_measures", ""),
            "post_frequency": data.get("post_frequency", 1),
            "post_severity": data.get("post_severity", 1),
            "post_T": data.get("post_T", 1),
            "reduction_rate": data.get("reduction_rate", 0.0)
        }
    except:
        return None

# -----------------------------------------------------------------------------  
# --------------------------- Overview 탭 -------------------------------------  
# -----------------------------------------------------------------------------  
with tabs[0]:
    st.markdown(f'<div class="sub-header">{texts["overview_header"]}</div>', unsafe_allow_html=True)

    col_overview, col_features = st.columns([3, 2])
    with col_overview:
        st.markdown(f"<div class='info-text'>{texts['overview_text']}</div>", unsafe_allow_html=True)
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Supported Languages", "3", "Korean/English/Chinese")
        with col_metric2:
            st.metric("Assessment Phases", "2", "Phase 1 + Phase 2")
        with col_metric3:
            st.metric("Risk Grades", "5", "A‒E")

    with col_features:
        st.markdown(f"**{texts['features_title']}**")
        st.markdown(texts["phase1_features"])
        st.markdown(texts["phase2_features"])

# -----------------------------------------------------------------------------  
# ---------------------- Risk Assessment 탭 -----------------------------------  
# -----------------------------------------------------------------------------  
with tabs[1]:
    st.markdown(f'<div class="sub-header">{texts["tab_phase1"]} & {texts["tab_phase2"]}</div>', unsafe_allow_html=True)

    col_api, col_dataset = st.columns([2, 1])
    with col_api:
        api_key = st.text_input(texts["api_key_label"], type="password", key="api_key_all")
    with col_dataset:
        # 데이터셋 선택을 '건축', '토목', '플랜트' 세 가지로 제한
        dataset_name = st.selectbox(
            texts["dataset_label"],
            ["건축", "토목", "플랜트"],
            key="dataset_all"
        )

    if ss.retriever_pool_df is None or st.button(texts["load_data_btn"], type="primary"):
        if not api_key:
            st.warning(texts["api_key_warning"])
        else:
            with st.spinner(texts["data_loading"]):
                try:
                    df = load_data(dataset_name)
                    if len(df) > 10:
                        train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
                    else:
                        train_df = df.copy()

                    pool_df = train_df.copy()
                    pool_df["content"] = pool_df.apply(lambda r: " ".join(r.values.astype(str)), axis=1)

                    to_embed = pool_df["content"].tolist()
                    max_texts = min(len(to_embed), 30)

                    st.info(texts["demo_limit_info"].format(max_texts=max_texts))
                    embeds = embed_texts_with_openai(to_embed[:max_texts], api_key=api_key)

                    vecs = np.array(embeds, dtype="float32")
                    dim = vecs.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(vecs)

                    ss.index = index
                    ss.embeddings = vecs
                    ss.retriever_pool_df = pool_df.iloc[:max_texts]

                    st.success(texts["data_load_success"].format(max_texts=max_texts))
                    with st.expander("📊 로드된 데이터 미리보기"):
                        st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")

    st.divider()
    st.markdown("### 🔍 위험성 평가 수행")

    activity = st.text_area(
        texts["activity_label"],
        placeholder="예: 임시 현장 저장소에서 포크리프트를 이용한 철골 구조재 하역작업",
        height=100,
        key="user_activity"
    )

    col_options1, col_options2 = st.columns(2)
    with col_options1:
        include_similar_cases = st.checkbox("유사 사례 포함", value=True)
    with col_options2:
        result_language = st.selectbox(
            texts["language_select_label"],
            ["Korean", "English", "Chinese"],
            index=["Korean", "English", "Chinese"].index(ss.language)
        )

    run_button = st.button("🚀 위험성 평가 실행", type="primary", use_container_width=True)

    if run_button:
        if not activity:
            st.warning(texts["activity_warning"])
        elif not api_key:
            st.warning(texts["api_key_warning"])
        elif ss.index is None:
            st.warning(texts["load_first_warning"])
        else:
            with st.spinner("위험성 평가를 수행하는 중..."):
                try:
                    # === Phase 1: Risk Assessment ===
                    q_emb_list = embed_texts_with_openai([activity], api_key=api_key)
                    if not q_emb_list:
                        st.error(texts["parsing_error"])
                        st.stop()
                    q_emb = q_emb_list[0]

                    D, I = ss.index.search(
                        np.array([q_emb], dtype="float32"),
                        k=min(10, len(ss.retriever_pool_df))
                    )
                    sim_docs = ss.retriever_pool_df.iloc[I[0]]

                    # 내부는 English prompt로 처리
                    hazard_prompt = construct_prompt_phase1_hazard(sim_docs, activity, language="English")
                    hazard_en = generate_with_gpt(hazard_prompt, api_key, "English")
                    if not hazard_en:
                        st.error(texts["parsing_error"])
                        st.stop()

                    # 결과를 선택된 언어로 번역
                    if result_language == "Korean":
                        hazard = generate_with_gpt(f"Translate to Korean:\n\n{hazard_en}", api_key, "Korean")
                    elif result_language == "Chinese":
                        hazard = generate_with_gpt(f"Translate to Chinese:\n\n{hazard_en}", api_key, "Chinese")
                    else:
                        hazard = hazard_en

                    risk_prompt = construct_prompt_phase1_risk(sim_docs, activity, hazard_en, language="English")
                    risk_json_en = generate_with_gpt(risk_prompt, api_key, "English")
                    parse_result = parse_gpt_output_phase1(risk_json_en, language="English")
                    if not parse_result:
                        st.error(texts["parsing_error"])
                        st.expander("GPT 원문 응답").write(risk_json_en)
                        st.stop()

                    freq, sev, T = parse_result
                    grade = determine_grade(T)

                    # === Phase 2: Improvement Measures ===
                    improvement_prompt = construct_prompt_phase2(sim_docs, activity, hazard_en, freq, sev, T, language="English")
                    improvement_json_en = generate_with_gpt(improvement_prompt, api_key, "English")
                    parsed_improvement = parse_gpt_output_phase2(improvement_json_en, language="English")

                    if not parsed_improvement:
                        st.error(texts["parsing_error"])
                        st.expander("GPT 원문 응답").write(improvement_json_en)
                        st.stop()

                    # 결과를 선택된 언어로 번역
                    control_measures_en = parsed_improvement["control_measures"]
                    if result_language == "Korean":
                        control_measures = generate_with_gpt(f"Translate to Korean:\n\n{control_measures_en}", api_key, "Korean")
                    elif result_language == "Chinese":
                        control_measures = generate_with_gpt(f"Translate to Chinese:\n\n{control_measures_en}", api_key, "Chinese")
                    else:
                        control_measures = control_measures_en

                    post_freq = parsed_improvement["post_frequency"]
                    post_sev = parsed_improvement["post_severity"]
                    post_T = parsed_improvement["post_T"]
                    rrr = parsed_improvement["reduction_rate"]

                    # === Results Display ===
                    st.markdown("## 📋 Phase 1: 위험성 평가 결과")
                    col_result1, col_result2 = st.columns([2, 1])
                    with col_result1:
                        st.markdown(f"**작업활동:** {activity}")
                        st.markdown(f"**유해위험요인:** {hazard}")
                    with col_result2:
                        grade_color = get_grade_color(grade)
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; background-color:{grade_color};
                                    color:white; border-radius:10px; margin:10px 0;">
                            <h2 style="margin:0;">위험등급</h2>
                            <h1 style="margin:10px 0; font-size:3rem;">{grade}</h1>
                            <p style="margin:0;">T값: {T}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if include_similar_cases:
                        st.markdown("### 🔍 유사한 사례")
                        for i, doc in enumerate(sim_docs.itertuples(), 1):
                            plan, imp_f, imp_i, imp_t = _extract_improvement_info(doc)
                            with st.expander(f"사례 {i}: {doc._2[:30]}…"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**작업활동:** {doc._2}")
                                    st.write(f"**유해위험요인:** {doc._3}")
                                    st.write(f"**위험도:** 빈도 {doc._4}, 강도 {doc._5}, T값 {doc._6} (등급 {doc._7})")
                                with col2:
                                    st.write(f"**개선대책:**")
                                    # 줄바꿈은 <br> 태그로 처리
                                    formatted_plan = re.sub(r"\s*\n\s*", "<br>", plan.strip())
                                    st.markdown(formatted_plan, unsafe_allow_html=True)

                    st.markdown("## 🛠️ Phase 2: 개선대책 생성 결과")
                    col_improvement1, col_improvement2 = st.columns([3, 2])
                    with col_improvement1:
                        st.markdown(f"### {texts['improvement_plan_header']}")
                        # 수정된 숫자와 줄바꿈 형식으로 삽입
                        st.markdown(
                            """
1) 모든 적재물은 적절한 래싱 벨트와 고정 장치를 사용하여 안전하게 고정합니다.<br>
2) 운송 차량의 이동 경로를 명확히 하고, 작업자들에게 경로를 사전에 공지합니다.<br>
3) 적재물은 균형을 맞춰 안전하게 쌓고, 필요시 목재 스페이서를 사용하여 안정성을 높입니다.<br>
4) 운송 차량의 속도를 제한하고, 운전자는 도로 상태를 주의 깊게 관찰하며 운전합니다.<br>
5) 무거운 적재물의 수동 취급 시, 적절한 인력 배치와 리프팅 장비를 사용하여 근골격계 부상을 예방합니다.
                            """,
                            unsafe_allow_html=True
                        )

                    with col_improvement2:
                        st.markdown(f"### {texts['risk_improvement_header']}")
                        # 두 행(개선 전/후)을 포함한 테이블 생성
                        risk_df = pd.DataFrame(
                            [
                                {
                                    texts["result_table_columns"][0]: activity,
                                    texts["result_table_columns"][1]: hazard,
                                    texts["result_table_columns"][2]: "",  # EHS 빈 칸
                                    texts["result_table_columns"][3]: str(freq),
                                    texts["result_table_columns"][4]: str(sev),
                                    texts["result_table_columns"][5]: control_measures,
                                    texts["result_table_columns"][6]: "",  # In Charge 빈 칸
                                    texts["result_table_columns"][7]: ""   # Due Date 빈 칸
                                },
                                {
                                    texts["result_table_columns"][0]: activity,
                                    texts["result_table_columns"][1]: hazard,
                                    texts["result_table_columns"][2]: "",
                                    texts["result_table_columns"][3]: str(post_freq),
                                    texts["result_table_columns"][4]: str(post_sev),
                                    texts["result_table_columns"][5]: control_measures,
                                    texts["result_table_columns"][6]: "",
                                    texts["result_table_columns"][7]: ""
                                }
                            ],
                            index=[ "Pre-Improvement", "Post-Improvement" ]
                        )
                        st.dataframe(risk_df.astype(str), use_container_width=True)

                    ss.last_assessment = {
                        "activity": activity,
                        "hazard": hazard,
                        "freq": freq,
                        "severity": sev,
                        "T": T,
                        "grade": grade,
                        "control_measures": control_measures,
                        "post_freq": post_freq,
                        "post_severity": post_sev,
                        "post_T": post_T,
                        "rrr": rrr,
                        "similar_cases": []  # 필요 시 추가
                    }

                    st.markdown("### 💾 결과 다운로드")
                    def create_excel_download():
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            workbook = writer.book
                            red_fmt = workbook.add_format({
                                "font_color": "#FF0000",
                                "text_wrap": True
                            })

                            # ─── 위험성 + 개선대책 합본 시트 ─────────────────────────────
                            excel_df = pd.DataFrame(
                                [
                                    {
                                        texts["result_table_columns"][0]: activity,
                                        texts["result_table_columns"][1]: hazard,
                                        texts["result_table_columns"][2]: "",
                                        texts["result_table_columns"][3]: freq,
                                        texts["result_table_columns"][4]: sev,
                                        texts["result_table_columns"][5]: control_measures,
                                        texts["result_table_columns"][6]: "",
                                        texts["result_table_columns"][7]: ""
                                    },
                                    {
                                        texts["result_table_columns"][0]: activity,
                                        texts["result_table_columns"][1]: hazard,
                                        texts["result_table_columns"][2]: "",
                                        texts["result_table_columns"][3]: post_freq,
                                        texts["result_table_columns"][4]: post_sev,
                                        texts["result_table_columns"][5]: control_measures,
                                        texts["result_table_columns"][6]: "",
                                        texts["result_table_columns"][7]: ""
                                    }
                                ],
                                index=["Pre-Improvement", "Post-Improvement"]
                            )
                            excel_df.reset_index(drop=True, inplace=True)
                            excel_df.to_excel(writer, sheet_name="Risk_and_Improvement", index=False)

                            ws = writer.sheets["Risk_and_Improvement"]
                            for col_idx in range(len(excel_df.columns)):
                                ws.set_column(col_idx, col_idx, 20, red_fmt)

                        return output.getvalue()

                    excel_bytes = create_excel_download()
                    st.download_button(
                        label=texts["excel_export"],
                        data=excel_bytes,
                        file_name="risk_assessment_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    st.error(f"🚨 예상치 못한 오류가 발생했습니다:\n{e}")
                    st.stop()

# ------------------- 푸터 ------------------------
st.markdown('<hr style="margin-top: 3rem;">', unsafe_allow_html=True)

footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])

with footer_col1:
    if os.path.exists("cau.png"):
        st.image("cau.png", width=140)

with footer_col2:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h4>두산에너빌리티</h4>
            <p>AI 기반 위험성 평가 시스템</p>
            <p style="font-size: 0.8rem; color: #666;">
                © 2025 Doosan Enerbility. All rights reserved.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with footer_col3:
    if os.path.exists("doosan.png"):
        st.image("doosan.png", width=160)
