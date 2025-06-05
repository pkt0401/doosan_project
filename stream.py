import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import io
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ----------------- 시스템 다국어 텍스트 -----------------
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_assessment": "위험성 평가 & 개선대책",
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
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "activity_warning": "작업활동을 입력하세요.",
        "include_similar": "유사 사례 포함",
        "result_language_label": "결과 언어 선택:",
        "run_button": "🚀 위험성 평가 실행",
        "phase1_header": "## 📋 Phase 1: 위험성 평가 결과",
        "phase2_header": "## 🛠️ Phase 2: 개선대책 생성 결과",
        "improvement_plan_header": "### 개선대책",
        "risk_improvement_header": "### 개선 전후 위험성 비교",
        "risk_table_pre": "Pre-Improvement",
        "risk_table_post": "Post-Improvement",
        "excel_export": "📥 결과 Excel 다운로드",
        "parsing_error": "위험성 평가 결과를 파싱할 수 없습니다.",
        "footer_text": "© 2025 Doosan Enerbility. All rights reserved."
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_assessment": "Risk Assessment & Improvement",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": "Doosan Enerbility AI Risk Assessment is an automated program trained on both on-demand risk-assessment reports from domestic and overseas construction sites and major-accident cases compiled by Korea's Ministry of Employment and Labor. Please ensure that every generated assessment is reviewed and approved by the On-Demand Risk Assessment Committee before it is used.",
        "features_title": "System Features and Components",
        "phase1_features": """
        #### Phase 1: Risk Assessment Automation
        - Learning risk assessment data according to work activities by process
        - Automatic hazard prediction when work activities are entered
        - Similar case search and display
        - Risk level (frequency, severity, T) measurement based on large language models (LLM)
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
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_btn": "Load Data and Configure Index",
        "api_key_warning": "Please enter an OpenAI API key to continue.",
        "data_loading": "Loading data and configuring index...",
        "demo_limit_info": "For demo purposes, only embedding {max_texts} texts. In a real environment, all data should be processed.",
        "data_load_success": "Data load and index configuration complete! (Total {max_texts} items processed)",
        "load_first_warning": "Please click the [Load Data and Configure Index] button first.",
        "activity_label": "Work Activity:",
        "activity_warning": "Please enter a work activity.",
        "include_similar": "Include Similar Cases",
        "result_language_label": "Select Result Language:",
        "run_button": "🚀 Run Risk Assessment",
        "phase1_header": "## 📋 Phase 1: Risk Assessment Results",
        "phase2_header": "## 🛠️ Phase 2: Improvement Measures Results",
        "improvement_plan_header": "### Control Measures",
        "risk_improvement_header": "### Pre/Post-Improvement Risk Comparison",
        "risk_table_pre": "Pre-Improvement",
        "risk_table_post": "Post-Improvement",
        "excel_export": "📥 Download Results as Excel",
        "parsing_error": "Unable to parse risk assessment results.",
        "footer_text": "© 2025 Doosan Enerbility. All rights reserved."
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_assessment": "风险评估 & 改进",
        "overview_header": "基于LLM的风险评估系统",
        "overview_text": "Doosan Enerbility AI 风险评估系统是一款自动化风险评估程序，基于国内外施工现场的'临时风险评估'数据以及韩国雇佣劳动部的重大事故案例进行训练开发而成。生成的风险评估结果必须经过临时风险评估审议委员会的审核后方可使用。",
        "features_title": "系统特点和组件",
        "phase1_features": """
        #### 第1阶段：风险评估自动化
        - 按工序学习与工作活动相关的风险评估数据
        - 输入工作活动时自动预测危害因素
        - 相似案例搜索和显示
        - 基于大型语言模型(LLM)的风险等级（频率、严重度、T值）测量
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
        "api_key_label": "输入OpenAI API密钥：",
        "dataset_label": "选择数据集",
        "load_data_btn": "加载数据和配置索引",
        "api_key_warning": "请输入OpenAI API密钥以继续。",
        "data_loading": "正在加载数据和配置索引...",
        "demo_limit_info": "出于演示目的，仅嵌入{max_texts}个文本。在实际环境中，应处理所有数据。",
        "data_load_success": "数据加载和索引配置完成！（共处理{max_texts}个项目）",
        "load_first_warning": "请先点击[加载数据和配置索引]按钮。",
        "activity_label": "工作活动：",
        "activity_warning": "请输入工作活动。",
        "include_similar": "包含相似案例",
        "result_language_label": "选择结果语言：",
        "run_button": "🚀 运行风险评估",
        "phase1_header": "## 📋 第1阶段：风险评估结果",
        "phase2_header": "## 🛠️ 第2阶段：改进措施结果",
        "improvement_plan_header": "### 控制措施",
        "risk_improvement_header": "### 改进前后风险比较",
        "risk_table_pre": "改进前",
        "risk_table_post": "改进后",
        "excel_export": "📥 下载结果为Excel",
        "parsing_error": "无法解析风险评估结果。",
        "footer_text": "© 2025 Doosan Enerbility. 版权所有。"
    }
}

# ----------------- 페이지 스타일 -----------------
st.set_page_config(page_title="AI Risk Assessment", page_icon="🛠️", layout="wide")
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
    "index": None,
    "embeddings": None,
    "retriever_pool_df": None,
    "last_assessment": None
}.items():
    if key not in ss:
        ss[key] = default

# ----------------- 언어 선택 UI -----------------
# 결과 언어를 하나만 선택하도록 함
result_language = st.selectbox(
    "결과 언어 선택:",
    ["Korean", "English", "Chinese"],
    index=0
)
texts = system_texts[result_language]

# ----------------- 헤더 -----------------
st.markdown(f'<div class="main-header">{texts["title"]}</div>', unsafe_allow_html=True)

# ----------------- 탭 구성 -----------------
tabs = st.tabs([texts["tab_overview"], texts["tab_assessment"]])

# -----------------------------------------------------------------------------  
# --------------------------- Overview 탭 -------------------------------------  
# -----------------------------------------------------------------------------  
with tabs[0]:
    st.markdown(f'<div class="sub-header">{texts["overview_header"]}</div>', unsafe_allow_html=True)

    col_overview, col_features = st.columns([3, 2])
    with col_overview:
        st.markdown(f"<div class='info-text'>{texts['overview_text']}</div>", unsafe_allow_html=True)
    with col_features:
        st.markdown(f"**{texts['features_title']}**")
        st.markdown(texts["phase1_features"])
        st.markdown(texts["phase2_features"])

# -----------------------------------------------------------------------------  
# ---------------------- Risk Assessment & Improvement 탭 ----------------------  
# -----------------------------------------------------------------------------  
with tabs[1]:
    st.markdown(f'<div class="sub-header">{texts["tab_assessment"]}</div>', unsafe_allow_html=True)

    col_api, col_dataset = st.columns([2, 1])
    with col_api:
        api_key = st.text_input(texts["api_key_label"], type="password")
    with col_dataset:
        dataset_name = st.selectbox(
            texts["dataset_label"],
            ["건축", "토목", "플랜트"]
        )

    # --- 데이터 로드 및 인덱스 구성 버튼 ---
    if ss.retriever_pool_df is None or st.button(texts["load_data_btn"], type="primary"):
        if not api_key:
            st.warning(texts["api_key_warning"])
        else:
            with st.spinner(texts["data_loading"]):
                try:
                    # 데이터 불러오기
                    def load_data(name: str):
                        if os.path.exists(f"{name}.xlsx"):
                            df = pd.read_excel(f"{name}.xlsx")
                        else:
                            # 샘플 데이터
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
                                "빈도": [3, 3, 2, 4, 2],
                                "강도": [5, 4, 5, 5, 3],
                                "개선대책": [
                                    "1) 안전벨트를 사용하여 하역물 고정 2) 적재물 균형 맞추기 3) 작업자 안전 교육 실시",
                                    "1) 비계 설치 및 안전대 사용 2) 추락 방지망 설치 3) 작업 전 점검",
                                    "1) 사면 경사 유지 2) 굴착 벽 보강 3) 지반 상태 점검",
                                    "1) 안전대 착용 의무화 2) 추락 방지망 설치 3) 작업 전 안전 교육",
                                    "1) 용접 부위 차단 2) 환기 시스템 사용 3) 보호구 착용"
                                ]
                            }
                            df = pd.DataFrame(data)
                        df["T"] = df["빈도"] * df["강도"]
                        def determine_grade(val):
                            if 16 <= val <= 25: return "A"
                            if 10 <= val <= 15: return "B"
                            if 5 <= val <= 9: return "C"
                            if 3 <= val <= 4: return "D"
                            if 1 <= val <= 2: return "E"
                            return "Unknown"
                        df["등급"] = df["T"].apply(determine_grade)
                        return df

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

                    # 임베딩 생성
                    def embed_texts(texts_list, api_key, model="text-embedding-3-large"):
                        client = OpenAI(api_key=api_key)
                        embeds = []
                        batch_size = 10
                        for i in range(0, len(texts_list), batch_size):
                            batch = texts_list[i : i + batch_size]
                            proc = [str(x).replace("\n", " ").strip() for x in batch]
                            try:
                                resp = client.embeddings.create(model=model, input=proc)
                                for item in resp.data:
                                    embeds.append(item.embedding)
                            except Exception as e:
                                st.error(f"임베딩 호출 실패 (배치 {i}): {e}")
                                for _ in batch:
                                    embeds.append([0.0] * 1536)
                        return embeds

                    embeds = embed_texts(to_embed[:max_texts], api_key)

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
        height=100
    )
    include_similar_cases = st.checkbox(texts["include_similar"], value=True)
    run_button = st.button(texts["run_button"], type="primary", use_container_width=True)

    if run_button:
        if not activity:
            st.warning(texts["activity_warning"])
        elif not api_key:
            st.warning(texts["api_key_warning"])
        elif ss.index is None:
            st.warning(texts["load_first_warning"])
        else:
            with st.spinner("처리 중..."):
                try:
                    # === Phase 1: Risk Assessment ===
                    client = OpenAI(api_key=api_key)

                    # Query embedding
                    def embed_single(text):
                        resp = client.embeddings.create(model="text-embedding-3-large", input=[text])
                        return resp.data[0].embedding

                    q_emb = embed_single(activity)
                    D, I = ss.index.search(np.array([q_emb], dtype="float32"), k=min(10, len(ss.retriever_pool_df)))
                    sim_docs = ss.retriever_pool_df.iloc[I[0]]

                    # 1) Hazard prediction prompt (English internal)
                    def construct_hazard_prompt(docs, activity):
                        prompt = "Here are examples of work activities and associated hazardous factors:\n\n"
                        for i, row in docs.head(5).iterrows():
                            prompt += f"Case {i+1}:\n- Work Activity: {row['작업활동 및 내용']}\n- Hazardous Factors: {row['유해위험요인 및 환경측면 영향']}\n\n"
                        prompt += f"Based on the above examples, predict the main hazardous factors for the following work activity:\n\nWork Activity: {activity}\n\nPredicted Hazardous Factors: "
                        return prompt

                    hazard_prompt = construct_hazard_prompt(sim_docs, activity)
                    hazard_en = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "You are a construction site risk assessment expert. Provide practical responses in English."},
                                  {"role": "user", "content": hazard_prompt}],
                        temperature=0.1,
                        max_tokens=200
                    ).choices[0].message.content.strip()

                    # Translate hazard to selected language if needed
                    if result_language == "Korean":
                        hazard = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "Translate English to Korean. Keep technical terms."},
                                      {"role": "user", "content": hazard_en}],
                            temperature=0.1,
                            max_tokens=200
                        ).choices[0].message.content.strip()
                    elif result_language == "Chinese":
                        hazard = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "Translate English to Chinese. Keep technical terms."},
                                      {"role": "user", "content": hazard_en}],
                            temperature=0.1,
                            max_tokens=200
                        ).choices[0].message.content.strip()
                    else:
                        hazard = hazard_en

                    # 2) Risk assessment prompt (English)
                    def construct_risk_prompt(docs, activity, hazard_en):
                        prompt = (
                            "Construction site risk assessment criteria:\n"
                            "- Frequency (1-5): 1=Very Rare, 2=Rare, 3=Occasional, 4=Frequent, 5=Very Frequent\n"
                            "- Severity (1-5): 1=Minor Injury, 2=Light Injury, 3=Moderate Injury, 4=Serious Injury, 5=Fatality\n"
                            "- T-value = Frequency × Severity\n\n"
                            "Reference cases:\n\n"
                        )
                        for i, row in docs.head(3).iterrows():
                            inp = f"{row['작업활동 및 내용']} - {row['유해위험요인 및 환경측면 영향']}"
                            freq = int(row["빈도"])
                            sev = int(row["강도"])
                            t_val = freq * sev
                            prompt += f"Case {i+1}:\nInput: {inp}\nAssessment: Frequency={freq}, Severity={sev}, T-value={t_val}\n\n"
                        prompt += (
                            f"Based on the above criteria and cases, assess the following:\n\n"
                            f"Work Activity: {activity}\n"
                            f"Hazardous Factors: {hazard_en}\n\n"
                            f"Respond in JSON format: " 
                            f'{{"frequency": number, "severity": number, "T": number}}'
                        )
                        return prompt

                    risk_prompt = construct_risk_prompt(sim_docs, activity, hazard_en)
                    risk_resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "You are a construction site risk assessment expert. Provide practical responses in English."},
                                  {"role": "user", "content": risk_prompt}],
                        temperature=0.1,
                        max_tokens=200
                    ).choices[0].message.content.strip()

                    # Parse JSON
                    match = re.search(r'\{"frequency":\s*([1-5]),\s*"severity":\s*([1-5]),\s*"T":\s*([0-9]+)\}', risk_resp)
                    if match:
                        freq = int(match.group(1))
                        sev = int(match.group(2))
                        T = int(match.group(3))
                    else:
                        nums = re.findall(r'\b([1-5])\b', risk_resp)
                        if len(nums) >= 2:
                            freq = int(nums[0])
                            sev = int(nums[1])
                            T = freq * sev
                        else:
                            st.error(texts["parsing_error"])
                            st.stop()

                    def determine_grade(val):
                        if 16 <= val <= 25: return "A"
                        if 10 <= val <= 15: return "B"
                        if 5 <= val <= 9: return "C"
                        if 3 <= val <= 4: return "D"
                        if 1 <= val <= 2: return "E"
                        return "Unknown"

                    grade = determine_grade(T)

                    # === Phase 2: Improvement Measures ===
                    def construct_improvement_prompt(docs, activity, hazard_en, freq, sev, T):
                        prompt = ""
                        for i, row in docs.head(3).iterrows():
                            plan = row.get("개선대책", "")
                            orig_f = int(row["빈도"])
                            orig_s = int(row["강도"])
                            orig_t = orig_f * orig_s
                            new_f = max(1, orig_f - 1)
                            new_s = max(1, orig_s - 1)
                            new_t = new_f * new_s
                            prompt += (
                                f"Example {i+1}:\n"
                                f"Input Work Activity: {row['작업활동 및 내용']}\n"
                                f"Input Hazardous Factors: {row['유해위험요인 및 환경측면 영향']}\n"
                                f"Input Original Frequency: {orig_f}\n"
                                f"Input Original Severity: {orig_s}\n"
                                f"Input Original T-value: {orig_t}\n"
                                f"Output (JSON):\n"
                                "{\n"
                                f'  "control_measures": "{plan}",\n'
                                f'  "post_frequency": {new_f},\n'
                                f'  "post_severity": {new_s},\n'
                                f'  "post_T": {new_t},\n'
                                f'  "reduction_rate": {((orig_t - new_t)/orig_t)*100:.2f}\n'
                                "}\n\n"
                            )
                        prompt += (
                            f"Now provide improvement measures in JSON for the following:\n\n"
                            f"Work Activity: {activity}\n"
                            f"Hazardous Factors: {hazard_en}\n"
                            f"Original Frequency: {freq}\n"
                            f"Original Severity: {sev}\n"
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

                    imp_prompt = construct_improvement_prompt(sim_docs, activity, hazard_en, freq, sev, T)
                    imp_resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "You are a construction site risk assessment expert. Provide practical responses in English."},
                                  {"role": "user", "content": imp_prompt}],
                        temperature=0.1,
                        max_tokens=300
                    ).choices[0].message.content.strip()

                    # Parse improvement JSON
                    imp_match = re.search(r'\{.*\}', imp_resp, re.DOTALL)
                    if not imp_match:
                        st.error(texts["parsing_error"])
                        st.stop()
                    import json
                    try:
                        imp_data = json.loads(imp_match.group())
                        ctrl_en = imp_data.get("control_measures", "")
                        post_freq = imp_data.get("post_frequency", 1)
                        post_sev = imp_data.get("post_severity", 1)
                        post_T = imp_data.get("post_T", post_freq * post_sev)
                        rrr = imp_data.get("reduction_rate", 0.0)
                    except:
                        st.error(texts["parsing_error"])
                        st.stop()

                    # Translate control measures if needed
                    if result_language == "Korean":
                        ctrl = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "Translate English to Korean. Keep technical terms."},
                                      {"role": "user", "content": ctrl_en}],
                            temperature=0.1,
                            max_tokens=200
                        ).choices[0].message.content.strip()
                    elif result_language == "Chinese":
                        ctrl = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "Translate English to Chinese. Keep technical terms."},
                                      {"role": "user", "content": ctrl_en}],
                            temperature=0.1,
                            max_tokens=200
                        ).choices[0].message.content.strip()
                    else:
                        ctrl = ctrl_en

                    # === Display Results ===
                    st.markdown(texts["phase1_header"])
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**작업활동 / Work Activity:** {activity}")
                        st.markdown(f"**유해위험요인 / Hazardous Factors:** {hazard}")
                        st.markdown(f"**빈도 / Frequency:** {freq}")
                        st.markdown(f"**강도 / Severity:** {sev}")
                        st.markdown(f"**T값 / T-value:** {T} (Grade: {grade})")
                    with col2:
                        color_map = {"A": "#ff1744","B": "#ff9800","C": "#ffc107","D": "#4caf50","E": "#2196f3"}
                        grade_color = color_map.get(grade, "#808080")
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; background-color:{grade_color};
                                    color:white; border-radius:10px; margin:10px 0;">
                            <h2 style="margin:0;">Grade</h2>
                            <h1 style="margin:10px 0; font-size:3rem;">{grade}</h1>
                            <p style="margin:0;">T-value: {T}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if include_similar_cases:
                        st.markdown("### 🔍 유사한 사례 / Similar Cases")
                        for i in range(len(sim_docs)):
                            doc = sim_docs.iloc[i]
                            plan_candidate, imp_f, imp_i, imp_t = "", max(1, int(doc["빈도"]) - 1), max(1, int(doc["강도"]) - 1), None
                            if "개선대책" in doc and pd.notna(doc["개선대책"]):
                                plan_candidate = doc["개선대책"]
                            imp_t = imp_f * imp_i
                            with st.expander(f"사례 {i+1}: {doc['작업활동 및 내용'][:30]}…"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.write(f"**작업활동 / Work Activity:** {doc['작업활동 및 내용']}")
                                    st.write(f"**유해위험요인 / Hazardous Factors:** {doc['유해위험요인 및 환경측면 영향']}")
                                    st.write(f"**빈도 / Frequency:** {doc['빈도']}")
                                    st.write(f"**강도 / Severity:** {doc['강도']}")
                                    st.write(f"**T값 / T-value:** {doc['T']} (Grade: {doc['등급']})")
                                with c2:
                                    st.write(f"**개선대책 / Control Measures:**")
                                    formatted = re.sub(r"\s*\n\s*", "<br>", plan_candidate.strip())
                                    st.markdown(formatted, unsafe_allow_html=True)

                    st.markdown(texts["phase2_header"])
                    col3, col4 = st.columns([3, 2])
                    with col3:
                        st.markdown(f"### {texts['improvement_plan_header']}")
                        # 고정된 개선대책 텍스트 예시 (줄바꿈 포함)
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
                    with col4:
                        st.markdown(f"### {texts['risk_improvement_header']}")
                        risk_df = pd.DataFrame(
                            [
                                {
                                    "Work Sequence": activity,
                                    "Hazardous Factors": hazard,
                                    "EHS": "",
                                    "Frequency": freq,
                                    "Severity": sev,
                                    "Control Measures": ctrl,
                                    "In Charge": "",
                                    "Correction Due Date": ""
                                },
                                {
                                    "Work Sequence": activity,
                                    "Hazardous Factors": hazard,
                                    "EHS": "",
                                    "Frequency": post_freq,
                                    "Severity": post_sev,
                                    "Control Measures": ctrl,
                                    "In Charge": "",
                                    "Correction Due Date": ""
                                }
                            ],
                            index=[texts["risk_table_pre"], texts["risk_table_post"]]
                        )
                        st.dataframe(risk_df.astype(str), use_container_width=True)

                    ss.last_assessment = {
                        "activity": activity,
                        "hazard": hazard,
                        "freq": freq,
                        "severity": sev,
                        "T": T,
                        "grade": grade,
                        "control_measures": ctrl,
                        "post_freq": post_freq,
                        "post_severity": post_sev,
                        "post_T": post_T,
                        "rrr": rrr
                    }

                    st.markdown("### 💾 결과 다운로드")
                    def create_excel_download():
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            workbook = writer.book
                            fmt = workbook.add_format({"font_color": "#FF0000", "text_wrap": True})

                            excel_df = pd.DataFrame(
                                [
                                    {
                                        "Work Sequence": activity,
                                        "Hazardous Factors": hazard,
                                        "EHS": "",
                                        "Frequency": freq,
                                        "Severity": sev,
                                        "Control Measures": ctrl,
                                        "In Charge": "",
                                        "Correction Due Date": ""
                                    },
                                    {
                                        "Work Sequence": activity,
                                        "Hazardous Factors": hazard,
                                        "EHS": "",
                                        "Frequency": post_freq,
                                        "Severity": post_sev,
                                        "Control Measures": ctrl,
                                        "In Charge": "",
                                        "Correction Due Date": ""
                                    }
                                ],
                                index=[texts["risk_table_pre"], texts["risk_table_post"]]
                            )
                            excel_df.reset_index(drop=True, inplace=True)
                            excel_df.to_excel(writer, sheet_name="Risk_and_Improvement", index=False)
                            ws = writer.sheets["Risk_and_Improvement"]
                            for col_idx in range(len(excel_df.columns)):
                                ws.set_column(col_idx, col_idx, 20, fmt)

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
with footer_col2:
    st.markdown(
        f"<div style='text-align: center; padding: 20px;'>{texts['footer_text']}</div>",
        unsafe_allow_html=True
    )
