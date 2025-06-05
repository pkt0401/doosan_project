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
        "tab_phase": "위험성 평가 & 개선대책",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": (
            "Doosan Enerbility AI Risk Assessment는 국내 및 해외 건설현장의 '수시 위험성 평가' 및 "
            "'노동부 중대재해 사례'를 학습하여 개발된 자동 위험성평가 프로그램입니다. "
            "생성된 위험성 평가는 반드시 수시 위험성평가 심의회를 통해 검증 후 사용하시기 바랍니다."
        ),
        "features_title": "시스템 특징 및 구성요소",
        "phase_features": (
            "#### Phase 1: 위험성 평가 자동화\n"
            "- 공정별 작업활동에 따른 위험성평가 데이터 학습\n"
            "- 작업활동 입력 시 유해위험요인 자동 예측 (영어 내부 실행)\n"
            "- 유사 사례 검색 및 표시 (영어 내부 처리 → 최종 출력 번역)\n"
            "- LLM 기반 위험도(빈도, 강도, T) 측정 (영어 내부 실행)\n"
            "- 위험등급(A–E) 자동 산정\n\n"
            "#### Phase 2: 개선대책 자동 생성\n"
            "- 맞춤형 개선대책 자동 생성 (영어 내부 실행)\n"
            "- 다국어(한국어/영어/중국어) 개선대책 생성 지원\n"
            "- 개선 전후 위험도(T) 자동 비교 분석\n"
            "- 공종/공정별 최적 개선대책 데이터베이스 구축"
        ),
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "include_similar_cases": "유사 사례 포함",
        "run_assessment": "🚀 위험성 평가 실행",
        "activity_warning": "작업활동을 입력하세요.",
        "performing_assessment": "위험성 평가를 수행하는 중...",
        "phase1_results": "📋 Phase 1: 위험성 평가 결과",
        "work_activity": "작업활동",
        "predicted_hazard": "예측된 유해위험요인",
        "risk_level_text": "위험도: 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})",
        "risk_grade_display": "위험등급",
        "t_value_display": "T값",
        "similar_cases_section": "🔍 유사한 사례",
        "case_number": "사례",
        "phase2_results": "🛠️ Phase 2: 개선대책 생성 결과",
        "improvement_plan_header": "개선대책",
        "risk_improvement_header": "위험도 개선 결과",
        "comparison_columns": ["항목", "개선 전", "개선 후"],
        "risk_reduction_label": "위험 감소율 (RRR)",
        "risk_visualization": "📊 위험도 변화 시각화",
        "before_improvement": "개선 전",
        "after_improvement": "개선 후",
        "grade_label": "등급",
        "download_results": "💾 결과 다운로드",
        "excel_export": "📥 결과 Excel 다운로드",
        # 엑셀 시트용 헤더 (한국어+영어 혼합) 
        "col_activity_header": "작업활동 및 내용 Work Sequence",
        "col_hazard_header": "유해위험요인 및 환경측면 영향 Hazarous Factors",
        "col_ehs_header": "EHS",
        "col_risk_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_risk_severity_header": "위험성 Risk – 강도 severity",
        "col_control_header": "개선대책 및 세부관리방안 Control Measures",
        "col_incharge_header": "개선담당자 In Charge",
        "col_duedate_header": "개선일자 Correction Due Date",
        "col_after_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_after_severity_header": "위험성 Risk – 강도 severity"
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_phase": "Risk Assessment & Improvement",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": (
            "Doosan Enerbility AI Risk Assessment is an automated program trained on on-demand risk-assessment reports "
            "from domestic and overseas construction sites and major-accident cases compiled by Korea's Ministry of Employment "
            "and Labor. Please ensure that every generated assessment is reviewed and approved by the On-Demand Risk Assessment "
            "Committee before it is used."
        ),
        "features_title": "System Features and Components",
        "phase_features": (
            "#### Phase 1: Risk Assessment Automation\n"
            "- Learning risk assessment data per work activity\n"
            "- Automatic hazard prediction when work activities are entered (internal: English)\n"
            "- Similar case search & display (internal: English → final: translation)\n"
            "- LLM-based risk level (frequency, intensity, T) measurement (internal: English)\n"
            "- Automatic risk grade (A–E) calculation\n\n"
            "#### Phase 2: Automatic Generation of Improvement Measures\n"
            "- Customized improvement measures generation (internal: English)\n"
            "- Multilingual (Korean/English/Chinese) improvement measure support\n"
            "- Automatic comparative analysis before/after improvement\n"
            "- Database of optimal improvement measures per process"
        ),
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_btn": "Load Data and Configure Index",
        "api_key_warning": "Please enter an OpenAI API key to continue.",
        "data_loading": "Loading data and configuring index...",
        "demo_limit_info": "For demo purposes, only embedding {max_texts} texts. In a real environment, process all data.",
        "data_load_success": "Data load and index configuration complete! (Total {max_texts} items processed)",
        "load_first_warning": "Please click [Load Data and Configure Index] first.",
        "activity_label": "Work Activity:",
        "include_similar_cases": "Include Similar Cases",
        "run_assessment": "🚀 Run Risk Assessment",
        "activity_warning": "Please enter a work activity.",
        "performing_assessment": "Performing risk assessment...",
        "phase1_results": "📋 Phase 1: Risk Assessment Results",
        "work_activity": "Work Activity",
        "predicted_hazard": "Predicted Hazard",
        "risk_level_text": "Risk Level: Frequency {freq}, Intensity {intensity}, T-value {t_value} (Grade {grade})",
        "risk_grade_display": "Risk Grade",
        "t_value_display": "T Value",
        "similar_cases_section": "🔍 Similar Cases",
        "case_number": "Case",
        "phase2_results": "🛠️ Phase 2: Improvement Measures Results",
        "improvement_plan_header": "Improvement Plan",
        "risk_improvement_header": "Risk Improvement Results",
        "comparison_columns": ["Item", "Before Improvement", "After Improvement"],
        "risk_reduction_label": "Risk Reduction Rate (RRR)",
        "risk_visualization": "📊 Risk Level Change Visualization",
        "before_improvement": "Before Improvement",
        "after_improvement": "After Improvement",
        "grade_label": "Grade",
        "download_results": "💾 Download Results",
        "excel_export": "📥 Download Excel Report",
        # 엑셀 시트용 헤더 (한글+영문 혼합)
        "col_activity_header": "작업활동 및 내용 Work Sequence",
        "col_hazard_header": "유해위험요인 및 환경측면 영향 Hazarous Factors",
        "col_ehs_header": "EHS",
        "col_risk_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_risk_severity_header": "위험성 Risk – 강도 severity",
        "col_control_header": "개선대책 및 세부관리방안 Control Measures",
        "col_incharge_header": "개선담당자 In Charge",
        "col_duedate_header": "개선일자 Correction Due Date",
        "col_after_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_after_severity_header": "위험성 Risk – 강도 severity"
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_phase": "风险评估 & 改进措施",
        "overview_header": "基于LLM的风险评估系统",
        "overview_text": (
            "Doosan Enerbility AI 风险评估系统是一款自动化风险评估程序，基于国内外施工现场的'临时风险评估'数据及韩国劳工部 "
            "重大事故案例训练开发而成。生成
