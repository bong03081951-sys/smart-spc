"""
스마트제조 공정분석 시스템 v2
공정능력분석(PCA) + 통계적공정관리(SPC) + 심층 진단 + 개선 비교
강의록 08_공정능력분석 / 09_통계적공정관리 기반
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io, datetime

from spc_functions import (
    calc_sigma_within, calc_sigma_overall,
    calc_cp_cpk, calc_pp_ppk, get_capability_grade,
    select_control_chart,
    calc_xbar_r_chart, calc_xbar_s_chart, calc_imr_chart,
    calc_attribute_chart,
    apply_nelson_rules, NELSON_RULE_DESC,
    remove_outliers_and_recalc,
    interpret_capability,
)

# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="스마트제조 공정분석 시스템",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color:#f4f6fb; }
.main-title {
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    color:white; padding:22px 32px; border-radius:14px; margin-bottom:20px;
}
.main-title h1 { color:white; margin:0; font-size:24px; }
.main-title p  { color:#a8b2d8; margin:4px 0 0 0; font-size:13px; }
.sec-hdr {
    font-size:17px; font-weight:700; color:#1a1a2e;
    border-left:4px solid #4C72B0; padding-left:10px;
    margin:22px 0 12px 0;
}
.kpi-card {
    background:white; border-radius:12px; padding:16px 18px;
    box-shadow:0 2px 8px rgba(0,0,0,0.07);
    text-align:center; border-top:5px solid #4C72B0; margin-bottom:10px;
}
.kpi-card.g  { border-top-color:#27ae60; }
.kpi-card.y  { border-top-color:#f39c12; }
.kpi-card.r  { border-top-color:#e74c3c; }
.kpi-label { font-size:12px; color:#6c757d; font-weight:600; }
.kpi-value { font-size:30px; font-weight:800; color:#2c3e50; margin:4px 0; }
.kpi-sub   { font-size:11px; color:#95a5a6; }
.kpi-badge { display:inline-block; padding:2px 10px; border-radius:20px;
             font-size:11px; font-weight:700; margin-top:5px; }
.bg  { background:#d4edda; color:#155724; }
.by  { background:#fff3cd; color:#856404; }
.br  { background:#f8d7da; color:#721c24; }
.verdict-card {
    border-radius:16px; padding:28px 32px; text-align:center;
    box-shadow:0 4px 16px rgba(0,0,0,0.10); margin-bottom:14px;
}
.verdict-card.stable  { background:linear-gradient(135deg,#d4edda,#a8d8b8); border:2px solid #27ae60; }
.verdict-card.caution { background:linear-gradient(135deg,#fff3cd,#fde68a); border:2px solid #f39c12; }
.verdict-card.danger  { background:linear-gradient(135deg,#f8d7da,#f5b7b1); border:2px solid #e74c3c; }
.verdict-icon  { font-size:52px; }
.verdict-title { font-size:26px; font-weight:900; margin:6px 0; }
.verdict-desc  { font-size:14px; color:#555; }
.ibox { border-radius:0 10px 10px 0; padding:13px 17px;
        margin:7px 0; font-size:14px; line-height:1.75; }
.ibox.info   { background:#eef2ff; border-left:4px solid #4C72B0; }
.ibox.ok     { background:#e8f5e9; border-left:4px solid #27ae60; }
.ibox.warn   { background:#fff8e1; border-left:4px solid #f39c12; }
.ibox.danger { background:#fde8e8; border-left:4px solid #e74c3c; }
.cause-card {
    background:white; border-radius:10px; padding:14px 16px;
    box-shadow:0 2px 6px rgba(0,0,0,0.07); margin-bottom:10px;
    border-left:4px solid #9b59b6;
}
.cause-title { font-weight:700; color:#6c3483; font-size:14px; }
.cause-body  { color:#555; font-size:13px; margin-top:5px; line-height:1.65; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════
def color_cls(v): return "g" if v>=1.33 else ("y" if v>=1.00 else "r")
def badge_cls(v): return "bg" if v>=1.33 else ("by" if v>=1.00 else "br")
def color_hex(v): return "#27ae60" if v>=1.33 else ("#f39c12" if v>=1.00 else "#e74c3c")

# ══════════════════════════════════════════════
# 샘플 데이터
# ══════════════════════════════════════════════
def get_sample_data(name):
    if name == "PVC 점도 (계량형, Xbar-R)":
        raw = np.array([
            [3576.27,3630.12,3576.27,3630.12,3355.69,3363.62],
            [3504.17,3514.52,3747.43,3666.15,3709.25,3317.28],
            [3440.11,3494.35,3962.93,3514.30,3273.57,3336.20],
            [3638.33,3719.84,3617.47,3450.17,3378.70,3475.50],
            [3661.94,3485.53,3499.43,3605.53,3390.29,3519.26],
        ])
        df = pd.DataFrame(raw, columns=[f'pl_{i}' for i in range(1,7)])
        return df.melt(var_name='prod_line', value_name='viscocity')
    elif name == "반도체 두께 (계량형, I-MR)":
        np.random.seed(42)
        return pd.DataFrame({'lot':range(1,31),'thickness':np.random.normal(40,2,30)})
    elif name == "LED 불량 (계수형, NP)":
        np.random.seed(7); n=50
        return pd.DataFrame({'Lot':range(1,n+1),'sample_size':[300]*n,
                             'Defectives':np.random.binomial(300,0.02,n)})
    elif name == "충전기 결함 (계수형, C)":
        np.random.seed(3); n=20
        return pd.DataFrame({'Lot':range(1,n+1),'sample_size':[500]*n,
                             'Defects':np.random.poisson(50,n)})

# ══════════════════════════════════════════════
# 차별화 기능 함수
# ══════════════════════════════════════════════
def overall_verdict(Cpk, nr_violations, nr_viol_count):
    cap_ok   = Cpk >= 1.00
    cap_good = Cpk >= 1.33
    spc_ok   = not nr_violations
    if cap_good and spc_ok:
        return "stable",  "🟢 안정",  "공정능력이 충분하고 관리 상태입니다."
    elif cap_ok and spc_ok:
        return "caution", "🟡 주의",  "공정능력은 만족하나 여유가 적습니다. 지속 모니터링이 필요합니다."
    elif cap_good and not spc_ok:
        return "caution", "🟡 주의",  "공정능력은 충분하나 SPC 이상이 감지됐습니다. 원인을 확인하세요."
    elif not cap_ok and not spc_ok:
        return "danger",  "🔴 위험",  f"공정능력 부족(Cpk={Cpk:.4f}) + SPC 이상 {nr_viol_count}건. 즉시 조치 필요."
    else:
        return "danger",  "🔴 위험",  f"공정능력 부족(Cpk={Cpk:.4f}). 공정 개선이 시급합니다."

def interpret_index_gaps(Cp, Cpk, Pp, Ppk, Cpu, Cpl):
    results = []
    cp_cpk = Cp - Cpk
    if cp_cpk < 0.05:
        results.append({"type":"ok","title":"✅ 중심 정렬 양호 (Cp ≈ Cpk)",
            "body":f"Cp={Cp:.4f}, Cpk={Cpk:.4f}, 차이={cp_cpk:.4f}<br>공정 평균이 규격 중심에 잘 맞춰져 있습니다.","loss_pct":0})
    else:
        direction = "USL(상한)" if Cpu < Cpl else "LSL(하한)"
        results.append({"type":"warn","title":f"⚠ 중심 편차 존재 — {direction} 방향 치우침",
            "body":f"Cp={Cp:.4f}, Cpk={Cpk:.4f}, 차이={cp_cpk:.4f}<br><b>중심편차 손실: {cp_cpk/Cp*100:.1f}%</b><br>"
                   f"치우침만 없애면 Cpk를 {Cp:.4f}까지 개선할 수 있습니다.",
            "loss_pct":round(cp_cpk/Cp*100,1)})
    cp_pp = Cp - Pp
    if cp_pp < 0.05:
        results.append({"type":"ok","title":"✅ 군간변동 안정 (Cp ≈ Pp)",
            "body":f"Cp={Cp:.4f}, Pp={Pp:.4f}, 차이={cp_pp:.4f}<br>장기적으로도 안정적입니다.","loss_pct":0})
    else:
        results.append({"type":"warn" if cp_pp<0.3 else "danger",
            "title":"⚠ 군간변동 큼 (Cp > Pp) — 장기 변동 존재",
            "body":f"Cp={Cp:.4f}(단기), Pp={Pp:.4f}(장기), 차이={cp_pp:.4f}<br><b>장기 성능 저하: {cp_pp/Cp*100:.1f}%</b><br>"
                   f"로트 간·교대 간·시간적 변동이 존재합니다.",
            "loss_pct":round(cp_pp/Cp*100,1)})
    return results

def diagnose_anomaly(nelson_result, cp_result, pp_result):
    violations = nelson_result['violations']
    Cp, Cpk    = cp_result['Cp'], cp_result['Cpk']
    Cpu, Cpl   = cp_result['Cpu'], cp_result['Cpl']
    diagnoses  = []
    if violations[2] or violations[3]:
        r_nums = []; detail = []
        if violations[2]: r_nums.append("Rule 2"); detail.append(f"연속 9점 편향 (부분군 {violations[2][:3]}...)")
        if violations[3]: r_nums.append("Rule 3"); detail.append(f"연속 6점 추세 (부분군 {violations[3][:3]}...)")
        diagnoses.append({"icon":"📈","cause":"공정 평균(μ) 이동","triggered":" + ".join(r_nums),
            "detail":"<br>".join(detail),
            "actions":["공정 파라미터(온도·압력·속도 등) 설정값 확인","원재료 로트 변경 여부 확인",
                       "작업자·교대조 변경 시점 확인","설비 마모·공구 마모 여부 확인"],
            "related_concept":"군간변동(between variance) 증가 → Cp > Pp 갭과 연결"})
    if violations[1] or violations[5]:
        r_nums = []; detail = []
        if violations[1]: r_nums.append("Rule 1"); detail.append(f"±3σ 이탈 (부분군 {violations[1][:5]})")
        if violations[5]: r_nums.append("Rule 5"); detail.append(f"±2σ 이탈 2/3점")
        diagnoses.append({"icon":"📊","cause":"공정 산포(σ) 증가","triggered":" + ".join(r_nums),
            "detail":"<br>".join(detail),
            "actions":["측정 시스템 재점검","원재료 균일성 확인","설비 진동 증가 여부 확인",
                       "4M(사람·기계·재료·방법) 변경 사항 점검"],
            "related_concept":"σ_within 증가 → Cp 저하로 직결"})
    if violations[1]:
        diagnoses.append({"icon":"🎯","cause":f"특정 부분군 이상 — {violations[1]}","triggered":"Rule 1",
            "detail":f"관리한계(±3σ) 이탈 부분군: {violations[1]}",
            "actions":[f"부분군 {violations[1]}의 생산기록·작업일지 확인",
                       "해당 시점 원재료 로트·설비 상태 확인",
                       "이상원인 제거 후 관리도 재작성","측정 오류 vs 실제 공정 이상 구분"],
            "related_concept":"이상원인(Assignable Cause) — 제거 가능한 변동"})
    if (Cp - Cpk) > 0.1:
        direction = "USL 방향(상한 초과 위험)" if Cpu < Cpl else "LSL 방향(하한 미달 위험)"
        diagnoses.append({"icon":"🎯","cause":f"규격 중심 이탈 — {direction}",
            "triggered":f"Cp({Cp:.3f}) - Cpk({Cpk:.3f}) = {Cp-Cpk:.3f} > 0.1",
            "detail":f"Cpu={Cpu:.4f}, Cpl={Cpl:.4f}",
            "actions":[f"공정 평균을 {'낮추는' if Cpu<Cpl else '높이는'} 방향으로 조정",
                       "목표값(Target)과 규격 중심 차이 재확인",
                       "Cp는 높고 Cpk가 낮으면 → 중심만 조정하면 됨"],
            "related_concept":"Cp ≠ Cpk → 중심 편차(Centering Loss)"})
    if violations[4] or violations[7] or violations[8]:
        diagnoses.append({"icon":"🔄","cause":"구조적 변동 패턴",
            "triggered":", ".join([f"Rule {r}" for r in [4,7,8] if violations[r]]),
            "detail":"<br>".join([f"Rule {r}: {NELSON_RULE_DESC[r]}" for r in [4,7,8] if violations[r]]),
            "actions":["Rule 4(진동): 과보정(over-adjustment) 여부 점검",
                       "Rule 7(층화): 부분군 재구성 검토","Rule 8(혼합): 두 개 이상 분포 혼합 가능성"],
            "related_concept":"계통적 패턴 → 이상원인(비연속적 변동)"})
    if not diagnoses:
        diagnoses.append({"icon":"✅","cause":"이상 원인 없음",
            "triggered":"모든 Nelson Rule 통과","detail":"현재 감지된 이상 패턴이 없습니다.",
            "actions":["현재 관리 상태를 유지하세요.","정기적인 공정 모니터링을 계속하세요."],
            "related_concept":"우연원인(Chance Cause)만 존재하는 관리 상태"})
    return diagnoses

def simulate_improvement(mu_before, sigma_before, USL, LSL, mu_after, sigma_after):
    def cp_cpk(mu, sw):
        Cp  = (USL-LSL)/(6*sw)
        Cpu = (USL-mu)/(3*sw); Cpl = (mu-LSL)/(3*sw)
        return Cp, min(Cpu,Cpl), Cpu, Cpl
    Cp_b,Cpk_b,Cpu_b,Cpl_b = cp_cpk(mu_before, sigma_before)
    Cp_a,Cpk_a,Cpu_a,Cpl_a = cp_cpk(mu_after,  sigma_after)
    return {
        "before":{"mu":mu_before,"sigma":sigma_before,"Cp":Cp_b,"Cpk":Cpk_b},
        "after": {"mu":mu_after, "sigma":sigma_after, "Cp":Cp_a,"Cpk":Cpk_a},
        "delta": {"Cp":Cp_a-Cp_b,"Cpk":Cpk_a-Cpk_b},
        "grade_before":get_capability_grade(Cpk_b),
        "grade_after": get_capability_grade(Cpk_a),
    }

# ══════════════════════════════════════════════
# 시각화 함수
# ══════════════════════════════════════════════
def plot_histogram(df, val_col, LSL, USL, mu, sw):
    vals = df[val_col].dropna()
    span = USL - LSL
    x_r  = np.linspace(min(vals.min(),LSL)-span*.1, max(vals.max(),USL)+span*.1, 400)
    y_r  = stats.norm.pdf(x_r, mu, sw)
    fig  = go.Figure()
    fig.add_trace(go.Histogram(x=vals,nbinsx=20,histnorm="probability density",
                               marker_color="#4C72B0",opacity=0.65,name="데이터"))
    fig.add_trace(go.Scatter(x=x_r,y=y_r,mode="lines",name="정규분포",
                             line=dict(color="#e74c3c",width=2.5)))
    for xv,lbl,col in [(LSL,"LSL","#e74c3c"),(USL,"USL","#e74c3c"),(mu,"μ","#27ae60")]:
        fig.add_vline(x=xv,line_dash="dash",line_color=col,line_width=2)
        fig.add_annotation(x=xv,y=0,yref="paper",text=f"<b>{lbl}</b>",
                           showarrow=False,yshift=12,font=dict(color=col,size=11))
    fig.update_layout(title="공정 분포 + 규격",xaxis_title=val_col,yaxis_title="밀도",
                      template="plotly_white",height=330,
                      legend=dict(orientation="h",y=-0.22),
                      margin=dict(l=40,r=20,t=45,b=40))
    return fig

def plot_qq(vals):
    z = stats.zscore(vals.dropna())
    (xq,yq),reg = stats.probplot(z,dist='norm')
    xl = np.array([-3,3]); yl = reg[0]*xl+reg[1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xq,y=yq,mode="markers",name="데이터",
                             marker=dict(color="#4C72B0",size=7,opacity=0.8)))
    fig.add_trace(go.Scatter(x=xl,y=yl,mode="lines",name="기준선",
                             line=dict(color="#e74c3c",width=2)))
    fig.update_layout(title="Q-Q Plot (정규성 확인)",
                      xaxis_title="이론적 분위수",yaxis_title="표본 분위수",
                      template="plotly_white",height=330,
                      margin=dict(l=40,r=20,t=45,b=40))
    return fig

def plot_boxplot(df, sg_col, val_col, LSL, USL):
    fig = px.box(df,x=sg_col,y=val_col,color=sg_col,points="all",
                 title="부분군별 분포",template="plotly_white")
    for yv,lbl in [(LSL,"LSL"),(USL,"USL")]:
        fig.add_hline(y=yv,line_dash="dash",line_color="#e74c3c",line_width=2,
                      annotation_text=f"<b>{lbl}</b>",annotation_position="right")
    fig.update_layout(height=330,showlegend=False,margin=dict(l=40,r=80,t=45,b=40))
    return fig

def plot_capability_bar(Cp, Cpk, Pp, Ppk):
    labels=["Cp","Cpk","Pp","Ppk"]; values=[Cp,Cpk,Pp,Ppk]
    fig = go.Figure()
    for lbl,val in zip(labels,values):
        fig.add_trace(go.Bar(name=lbl,x=[lbl],y=[val],marker_color=color_hex(val),
                             text=f"{val:.4f}",textposition="outside",width=0.5))
    for yv,txt,col in [(1.67,"1.67","#1abc9c"),(1.33,"1.33","#27ae60"),
                        (1.00,"1.00","#f39c12"),(0.67,"0.67","#e74c3c")]:
        fig.add_hline(y=yv,line_dash="dot",line_color=col,line_width=1.2,
                      annotation_text=txt,annotation_position="right",
                      annotation_font=dict(size=10,color=col))
    fig.update_layout(title="단기 vs 장기 공정능력 비교",
                      yaxis=dict(range=[0,max(max(values)*1.25,1.8)],title="지수 값"),
                      template="plotly_white",height=330,showlegend=False,
                      margin=dict(l=40,r=100,t=45,b=40),bargap=0.35)
    return fig

def plot_control_chart(chart_result, val_col, nelson_result=None):
    ct = chart_result['chart_type']
    if ct=='I-MR':
        p1,U1,C1,L1,n1=chart_result['I_point'],chart_result['I_UCL'],chart_result['I_CL'],chart_result['I_LCL'],"I"
        p2,U2,C2,L2,n2=chart_result['MR_point'],chart_result['MR_UCL'],chart_result['MR_CL'],chart_result['MR_LCL'],"MR"
    elif ct=='Xbar-R':
        p1,U1,C1,L1,n1=chart_result['Xbar_point'],chart_result['Xbar_UCL'],chart_result['Xbar_CL'],chart_result['Xbar_LCL'],"Xbar"
        p2,U2,C2,L2,n2=chart_result['R_point'],chart_result['R_UCL'],chart_result['R_CL'],chart_result['R_LCL'],"R"
    else:
        p1,U1,C1,L1,n1=chart_result['Xbar_point'],chart_result['Xbar_UCL'],chart_result['Xbar_CL'],chart_result['Xbar_LCL'],"Xbar"
        p2,U2,C2,L2,n2=chart_result['S_point'],chart_result['S_UCL'],chart_result['S_CL'],chart_result['S_LCL'],"S"
    viol_idx = set()
    if nelson_result:
        for idxs in nelson_result['violations'].values(): viol_idx.update(idxs)
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.12,
                        subplot_titles=[f"{n1} 관리도",f"{n2} 관리도"])
    def add_traces(fig, row, point, UCL, CL, LCL):
        x=point.index.tolist(); y=point.values.tolist()
        xok=[xi for xi in x if xi not in viol_idx]
        yok=[yi for xi,yi in zip(x,y) if xi not in viol_idx]
        xwn=[xi for xi in x if xi in viol_idx]
        ywn=[yi for xi,yi in zip(x,y) if xi in viol_idx]
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color="#4C72B0",width=1.5),showlegend=False),row=row,col=1)
        fig.add_trace(go.Scatter(x=xok,y=yok,mode="markers",name="정상",
                                 marker=dict(color="#4C72B0",size=8),showlegend=(row==1)),row=row,col=1)
        if xwn:
            fig.add_trace(go.Scatter(x=xwn,y=ywn,mode="markers",name="⚠ 이상점",
                                     marker=dict(color="#e74c3c",size=13,symbol="circle",
                                                 line=dict(color="white",width=2)),showlegend=(row==1)),row=row,col=1)
        for yv,lbl,col,dash in [(UCL,f"UCL={UCL:.3f}","#e74c3c","dot"),
                                  (CL,f"CL={CL:.3f}","#27ae60","dashdot"),
                                  (LCL,f"LCL={LCL:.3f}","#e74c3c","dot")]:
            fig.add_hline(y=yv,line_dash=dash,line_color=col,line_width=1.5,row=row,col=1)
    add_traces(fig,1,p1,U1,C1,L1)
    add_traces(fig,2,p2,U2,C2,L2)
    fig.update_layout(title=f"{ct} 관리도 — {val_col}",template="plotly_white",height=520,
                      margin=dict(l=50,r=60,t=65,b=40),legend=dict(orientation="h",y=-0.06))
    return fig

def plot_attribute_chart(chart_result, val_col, nelson_result=None):
    point=chart_result['point']; CL=chart_result['CL']
    fixed=chart_result['fixed_limits']
    viol_idx=set()
    if nelson_result:
        for idxs in nelson_result['violations'].values(): viol_idx.update(idxs)
    x=point.index.tolist(); y=point.values.tolist()
    UCL_v=[chart_result['UCL']]*len(x) if fixed else chart_result['UCL'].values.tolist()
    LCL_v=[chart_result['LCL']]*len(x) if fixed else chart_result['LCL'].values.tolist()
    xok=[xi for xi,yi,u,l in zip(x,y,UCL_v,LCL_v) if l<=yi<=u and xi not in viol_idx]
    yok=[yi for xi,yi,u,l in zip(x,y,UCL_v,LCL_v) if l<=yi<=u and xi not in viol_idx]
    xwn=[xi for xi,yi,u,l in zip(x,y,UCL_v,LCL_v) if yi>u or yi<l or xi in viol_idx]
    ywn=[yi for xi,yi,u,l in zip(x,y,UCL_v,LCL_v) if yi>u or yi<l or xi in viol_idx]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color="#4C72B0",width=1.5),showlegend=False))
    fig.add_trace(go.Scatter(x=xok,y=yok,mode="markers",name="정상",marker=dict(color="#4C72B0",size=8)))
    if xwn:
        fig.add_trace(go.Scatter(x=xwn,y=ywn,mode="markers",name="⚠ 이상점",
                                 marker=dict(color="#e74c3c",size=13,symbol="circle",
                                             line=dict(color="white",width=2))))
    if fixed:
        for yv,lbl,col,dash in [(UCL_v[0],f"UCL={UCL_v[0]:.4f}","#e74c3c","dot"),
                                  (CL,f"CL={CL:.6f}","#27ae60","dashdot"),
                                  (LCL_v[0],f"LCL={LCL_v[0]:.4f}","#e74c3c","dot")]:
            fig.add_hline(y=yv,line_dash=dash,line_color=col,line_width=1.5,
                          annotation_text=f"<b>{lbl}</b>",annotation_position="right",
                          annotation_font=dict(color=col,size=10))
    else:
        fig.add_trace(go.Scatter(x=x,y=UCL_v,mode="lines",name="UCL",line=dict(color="#e74c3c",dash="dot",width=1.5)))
        fig.add_trace(go.Scatter(x=x,y=LCL_v,mode="lines",name="LCL",line=dict(color="#e74c3c",dash="dot",width=1.5)))
        fig.add_hline(y=CL,line_dash="dashdot",line_color="#27ae60",line_width=1.5,
                      annotation_text=f"<b>CL={CL:.6f}</b>",annotation_position="right")
    fig.update_layout(title=f"{chart_result['chart_type']} 관리도 — {val_col}",
                      xaxis_title="Lot",yaxis_title=val_col,
                      template="plotly_white",height=380,
                      margin=dict(l=50,r=130,t=55,b=40),legend=dict(orientation="h",y=-0.18))
    return fig

def plot_rewrite_comparison(initial, final):
    ct=initial['chart_type']
    if ct=='I-MR': p1,p2=initial['I_point'],final['I_point']; u1,l1,cl1=initial['I_UCL'],initial['I_LCL'],initial['I_CL']; u2,l2,cl2=final['I_UCL'],final['I_LCL'],final['I_CL']
    else: p1,p2=initial['Xbar_point'],final['Xbar_point']; u1,l1,cl1=initial['Xbar_UCL'],initial['Xbar_LCL'],initial['Xbar_CL']; u2,l2,cl2=final['Xbar_UCL'],final['Xbar_LCL'],final['Xbar_CL']
    fig=make_subplots(rows=1,cols=2,subplot_titles=["① 초기 관리도","② 재작성 관리도"])
    for ci,(point,UCL,CL,LCL) in enumerate([(p1,u1,cl1,l1),(p2,u2,cl2,l2)],1):
        x=point.index.tolist(); y=point.values.tolist()
        c="#95a5a6" if ci==1 else "#4C72B0"
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",marker=dict(color=c,size=7),
                                 line=dict(color=c,width=1.5),showlegend=False),row=1,col=ci)
        for yv,lc,dash in [(UCL,"#e74c3c","dot"),(CL,"#27ae60","dashdot"),(LCL,"#e74c3c","dot")]:
            fig.add_hline(y=yv,line_dash=dash,line_color=lc,line_width=1.5,row=1,col=ci)
    fig.update_layout(template="plotly_white",height=320,margin=dict(l=40,r=40,t=55,b=40),
                      title="관리도 재작성 전/후 비교")
    return fig

def plot_improvement_dist(result, USL, LSL):
    x_r = np.linspace(LSL - abs(USL-LSL)*0.15, USL + abs(USL-LSL)*0.15, 500)
    fig = go.Figure()
    for key,label,color,dash in [("before","개선 전","#e74c3c","dash"),("after","개선 후","#27ae60","solid")]:
        mu,sw = result[key]["mu"],result[key]["sigma"]
        y = stats.norm.pdf(x_r, mu, sw)
        fig.add_trace(go.Scatter(x=x_r,y=y,mode="lines",name=label,
                                 line=dict(color=color,width=2.5,dash=dash),
                                 fill="tozeroy" if key=="after" else None,
                                 fillcolor="rgba(39,174,96,0.08)" if key=="after" else None))
        fig.add_vline(x=mu,line_dash="dot",line_color=color,line_width=1.5,
                      annotation_text=f"μ={mu:.2f}",annotation_font=dict(color=color))
    for xv,lbl in [(LSL,"LSL"),(USL,"USL")]:
        fig.add_vline(x=xv,line_dash="dash",line_color="#2c3e50",line_width=2)
        fig.add_annotation(x=xv,y=0,yref="paper",text=f"<b>{lbl}</b>",
                           showarrow=False,yshift=12,font=dict(color="#2c3e50"))
    fig.update_layout(title="개선 전/후 공정 분포 비교",xaxis_title="관측값",yaxis_title="확률밀도",
                      template="plotly_white",height=340,legend=dict(orientation="h",y=-0.2),
                      margin=dict(l=40,r=30,t=50,b=40))
    return fig

def plot_improvement_bar(result):
    metrics=["Cp","Cpk"]
    before=[result["before"][m] for m in metrics]
    after=[result["after"][m] for m in metrics]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="개선 전",x=metrics,y=before,marker_color="#e74c3c",opacity=0.8,
                         text=[f"{v:.4f}" for v in before],textposition="outside"))
    fig.add_trace(go.Bar(name="개선 후",x=metrics,y=after,marker_color="#27ae60",opacity=0.9,
                         text=[f"{v:.4f}" for v in after],textposition="outside"))
    for yv,lbl,col in [(1.33,"1.33","#27ae60"),(1.00,"1.00","#f39c12"),(0.67,"0.67","#e74c3c")]:
        fig.add_hline(y=yv,line_dash="dot",line_color=col,line_width=1.2,
                      annotation_text=lbl,annotation_position="right",annotation_font=dict(size=10,color=col))
    fig.update_layout(title="공정능력 지수 개선 전/후",barmode="group",template="plotly_white",height=340,
                      yaxis=dict(title="지수 값",range=[0,max(max(before),max(after))*1.28]),
                      legend=dict(orientation="h",y=-0.2),margin=dict(l=40,r=100,t=50,b=40))
    return fig

# ══════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 공정분석 시스템")
    st.markdown("---")
    st.markdown("### 📁 데이터 입력")
    data_source = st.radio("데이터 소스", ["샘플 데이터","파일 업로드","직접 입력"])
    df_raw = None

    if data_source == "샘플 데이터":
        sname = st.selectbox("샘플 선택",[
            "PVC 점도 (계량형, Xbar-R)",
            "반도체 두께 (계량형, I-MR)",
            "LED 불량 (계수형, NP)",
            "충전기 결함 (계수형, C)",
        ])
        df_raw = get_sample_data(sname)
    elif data_source == "파일 업로드":
        uploaded = st.file_uploader("CSV/Excel",type=["csv","xlsx","xls"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    try: df_raw = pd.read_csv(uploaded, encoding="utf-8")
                    except: df_raw = pd.read_csv(uploaded, encoding="euc-kr")
                else: df_raw = pd.read_excel(uploaded)
                st.success(f"✅ {len(df_raw)}행 로드")
            except Exception as e: st.error(str(e))
    else:
        txt = st.text_area("데이터 붙여넣기",height=160,placeholder="sg,value\n1,10.2\n1,9.8")
        if txt.strip():
            try: df_raw=pd.read_csv(io.StringIO(txt)); st.success(f"✅ {len(df_raw)}행")
            except Exception as e: st.error(str(e))

    st.markdown("---")
    if df_raw is not None:
        cols = df_raw.columns.tolist()
        st.markdown("### ⚙️ 분석 설정")
        data_type = st.radio("데이터 유형",["계량형 (연속)","계수형 (이산)"])
        is_cont   = (data_type == "계량형 (연속)")
        sg_col  = st.selectbox("부분군 컬럼", cols, index=0)
        val_col = st.selectbox("관측값 컬럼", cols, index=min(1,len(cols)-1))
        n_col, attr_type = None, None
        if not is_cont:
            n_col    = st.selectbox("표본 크기 컬럼",[c for c in cols if c not in [sg_col,val_col]])
            attr_type= st.radio("결함 유형",["불량 (Defective)","결함 (Defect)"])
        if is_cont:
            st.markdown("---")
            st.markdown("### 📐 규격 입력")
            spec_type = st.radio("규격 종류",["양측 규격","USL만","LSL만"],horizontal=True)
            val_center = float(df_raw[val_col].mean())
            val_range  = float(df_raw[val_col].std()) * 4
            LSL, USL = None, None
            c1,c2 = st.columns(2)
            if spec_type in ["양측 규격","LSL만"]:
                with c1: LSL = st.number_input("LSL",value=round(val_center-val_range/2,2),format="%.2f")
            if spec_type in ["양측 규격","USL만"]:
                with c2: USL = st.number_input("USL",value=round(val_center+val_range/2,2),format="%.2f")
            if LSL is None: LSL = USL - val_range*2
            if USL is None: USL = LSL + val_range*2
            target = st.number_input("목표값",value=round((LSL+USL)/2,2),format="%.2f")
            sm_sel = st.selectbox("σ_within 계산",["pooled","range","std"])
        st.markdown("---")
        st.markdown("### 📊 관리도 설정")
        window       = st.slider("I-MR 윈도우",2,5,3)
        show_rewrite = st.checkbox("이상치 재작성 표시",value=True)
        st.markdown("---")
        run_btn = st.button("🚀 분석 실행",type="primary",use_container_width=True)

# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
st.markdown("""
<div class="main-title">
  <h1>🏭 스마트제조 공정분석 시스템</h1>
  <p>공정능력분석(PCA) + 통계적공정관리(SPC) + 심층 진단 + 개선 비교</p>
</div>""", unsafe_allow_html=True)

if df_raw is None:
    st.info("👈 사이드바에서 데이터를 선택해 주세요."); st.stop()

# 사전 공통 계산
cp_res=pp_res=chart_result=nelson=None
chart_sel=None; ct=""; verdict_cls=verdict_label=verdict_desc=""

if is_cont:
    try:
        cp_res = calc_cp_cpk(df_raw,sg_col,val_col,USL,LSL,sm_sel)
        pp_res = calc_pp_ppk(df_raw,sg_col,val_col,USL,LSL)
        Cp,Cpk  = cp_res['Cp'], cp_res['Cpk']
        Pp,Ppk  = pp_res['Pp'], pp_res['Ppk']
        Cpu,Cpl = cp_res['Cpu'],cp_res['Cpl']
        mu,sw,so= cp_res['mu'],cp_res['sigma_within'],pp_res['sigma_overall']
        sg_sizes = df_raw.groupby(sg_col)[val_col].count()
        n_mode   = int(sg_sizes.mode().iloc[0])
        chart_sel= select_control_chart('continuous',n_mode)
        ct       = chart_sel['chart_type']
        if ct=='I-MR':
            chart_result=calc_imr_chart(df_raw,sg_col,val_col,window)
            pt_nr=chart_result['I_point']; u_nr,c_nr,l_nr=chart_result['I_UCL'],chart_result['I_CL'],chart_result['I_LCL']
        elif ct=='Xbar-R':
            chart_result=calc_xbar_r_chart(df_raw,sg_col,val_col)
            pt_nr=chart_result['Xbar_point']; u_nr,c_nr,l_nr=chart_result['Xbar_UCL'],chart_result['Xbar_CL'],chart_result['Xbar_LCL']
        else:
            chart_result=calc_xbar_s_chart(df_raw,sg_col,val_col)
            pt_nr=chart_result['Xbar_point']; u_nr,c_nr,l_nr=chart_result['Xbar_UCL'],chart_result['Xbar_CL'],chart_result['Xbar_LCL']
        nelson=apply_nelson_rules(pt_nr,u_nr,c_nr,l_nr)
        nr_viol_cnt=sum(len(v) for v in nelson['violations'].values())
        verdict_cls,verdict_label,verdict_desc=overall_verdict(Cpk,nelson['any_violation'],nr_viol_cnt)
    except Exception as e:
        st.error(f"계산 오류: {e}"); st.exception(e); st.stop()

# ══════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📋 데이터 개요",
    "📈 공정능력분석",
    "🎛 통계적공정관리",
    "🔍 심층 진단",
    "📐 개선 비교",
])

# ── TAB 1
with tab1:
    st.markdown('<div class="sec-hdr">📋 데이터 미리보기</div>',unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    m1.metric("총 행 수",f"{len(df_raw):,}"); m2.metric("컬럼 수",f"{len(df_raw.columns)}"); m3.metric("결측치",f"{df_raw.isnull().sum().sum()}")
    st.dataframe(df_raw.head(20),use_container_width=True,height=280)
    if is_cont and val_col in df_raw.columns:
        st.markdown('<div class="sec-hdr">📊 기초 통계량</div>',unsafe_allow_html=True)
        desc=df_raw[val_col].describe(); dc=st.columns(8)
        for i,(n,v) in enumerate(desc.items()):
            with dc[i%8]: st.metric(n,f"{v:.3f}")
        st.markdown('<div class="sec-hdr">🔬 정규성 검정 (Shapiro-Wilk)</div>',unsafe_allow_html=True)
        sv=df_raw[val_col].dropna()
        if len(sv)>5000: sv=sv.sample(5000,random_state=42)
        wstat,pval=stats.shapiro(sv)
        nc1,nc2,nc3=st.columns(3)
        nc1.metric("W 통계량",f"{wstat:.4f}"); nc2.metric("p-value",f"{pval:.4f}")
        with nc3:
            if pval>=0.05: st.success("✅ 정규성 만족")
            else: st.error("❌ 정규성 불만족")
        if pval<0.05: st.warning("⚠ 정규성 불만족 — Box-Cox 변환 등 전처리를 고려하세요.")

# ── TAB 2
with tab2:
    if not is_cont: st.info("공정능력분석은 계량형 데이터에만 적용됩니다."); st.stop()
    if cp_res is None: st.error("계산 오류"); st.stop()
    Cp=cp_res['Cp']; Cpk=cp_res['Cpk']
    Pp=pp_res['Pp']; Ppk=pp_res['Ppk']
    Cpu=cp_res['Cpu']; Cpl=cp_res['Cpl']
    mu=cp_res['mu']; sw=cp_res['sigma_within']; so=pp_res['sigma_overall']
    st.markdown('<div class="sec-hdr">📊 공정능력 지수</div>',unsafe_allow_html=True)
    kc=st.columns(4)
    for col_,(lbl,val,sub) in zip(kc,[("Cp",Cp,"단기 산포"),("Cpk",Cpk,"단기 종합"),
                                       ("Pp",Pp,"장기 산포"),("Ppk",Ppk,"장기 종합")]):
        g=get_capability_grade(val); cls_=color_cls(val); bc=badge_cls(val)
        with col_:
            st.markdown(f"""<div class="kpi-card {cls_}">
              <div class="kpi-label">{lbl}</div><div class="kpi-value">{val:.4f}</div>
              <div class="kpi-sub">{sub}</div>
              <span class="kpi-badge {bc}">{g['color']} 등급{g['grade']}: {g['status']}</span>
            </div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📐 σ 비교 — 단기 vs 장기</div>',unsafe_allow_html=True)
    sc=st.columns(4)
    gap_pct=(so**2-sw**2)/so**2*100 if so>0 else 0
    sc[0].metric("μ (평균)",f"{mu:.2f}"); sc[1].metric("σ_within (단기)",f"{sw:.4f}")
    sc[2].metric("σ_overall (장기)",f"{so:.4f}")
    sc[3].metric("Cp−Pp 갭",f"{Cp-Pp:.4f}",delta=f"군간변동 ≈{gap_pct:.1f}%",delta_color="inverse")
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📊 분포 시각화</div>',unsafe_allow_html=True)
    v1,v2=st.columns(2)
    with v1: st.plotly_chart(plot_histogram(df_raw,val_col,LSL,USL,mu,sw),use_container_width=True)
    with v2: st.plotly_chart(plot_qq(df_raw[val_col]),use_container_width=True)
    v3,v4=st.columns(2)
    with v3: st.plotly_chart(plot_boxplot(df_raw,sg_col,val_col,LSL,USL),use_container_width=True)
    with v4: st.plotly_chart(plot_capability_bar(Cp,Cpk,Pp,Ppk),use_container_width=True)
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🗒️ 공정능력 자동 해석</div>',unsafe_allow_html=True)
    for msg in interpret_capability(cp_res,pp_res):
        box_t="warn" if ("편차" in msg or "군간" in msg) else "ok"
        if "🔴" in msg: box_t="danger"
        st.markdown(f'<div class="ibox {box_t}">{msg}</div>',unsafe_allow_html=True)

# ── TAB 3
with tab3:
    try:
        if is_cont:
            spc_chart_sel=chart_sel; spc_ct=ct
        else:
            sg_sz2=df_raw.groupby(sg_col)[n_col].first(); nm2=int(sg_sz2.mode().iloc[0])
            vr2=(sg_sz2.std()>0); ak2="defective" if "불량" in attr_type else "defect"
            spc_chart_sel=select_control_chart('attribute',nm2,size_varies=vr2,attribute_type=ak2)
            spc_ct=spc_chart_sel['chart_type']
        st.markdown('<div class="sec-hdr">📌 관리도 자동 추천</div>',unsafe_allow_html=True)
        ic1,ic2=st.columns([1,2])
        with ic1:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">선택된 관리도</div>
              <div class="kpi-value" style="font-size:26px">{spc_ct}</div>
              <div class="kpi-sub">{spc_chart_sel['distribution']}</div>
            </div>""",unsafe_allow_html=True)
        with ic2:
            st.markdown(f"""<div class="ibox info">
            <b>🔍 선택 근거:</b> {spc_chart_sel['reason']}<br>
            <b>📐 분포 가정:</b> {spc_chart_sel['distribution']}<br>
            <b>📊 구성 차트:</b> {" + ".join(spc_chart_sel['charts'])}
            </div>""",unsafe_allow_html=True)
        with st.expander("🔧 관리도 수동 변경"):
            opts=["I-MR","Xbar-R","Xbar-S","NP","P","C","U"]
            spc_ct=st.selectbox("관리도 유형",opts,index=opts.index(spc_ct))
        st.markdown("---")
        if spc_ct in ("I-MR","Xbar-R","Xbar-S"):
            if spc_ct=="I-MR":
                cr=calc_imr_chart(df_raw,sg_col,val_col,window)
                pt_s,us,cs,ls=cr['I_point'],cr['I_UCL'],cr['I_CL'],cr['I_LCL']
            elif spc_ct=="Xbar-R":
                cr=calc_xbar_r_chart(df_raw,sg_col,val_col)
                pt_s,us,cs,ls=cr['Xbar_point'],cr['Xbar_UCL'],cr['Xbar_CL'],cr['Xbar_LCL']
            else:
                cr=calc_xbar_s_chart(df_raw,sg_col,val_col)
                pt_s,us,cs,ls=cr['Xbar_point'],cr['Xbar_UCL'],cr['Xbar_CL'],cr['Xbar_LCL']
            nr_s=apply_nelson_rules(pt_s,us,cs,ls)
            lc=st.columns(3); lc[0].metric("UCL",f"{us:.4f}"); lc[1].metric("CL",f"{cs:.4f}"); lc[2].metric("LCL",f"{ls:.4f}")
            st.markdown('<div class="sec-hdr">📈 관리도</div>',unsafe_allow_html=True)
            st.plotly_chart(plot_control_chart(cr,val_col,nr_s),use_container_width=True)
        else:
            cr=calc_attribute_chart(df_raw,sg_col,n_col,val_col,spc_ct)
            pt_s=cr['point']; cl_s=cr['CL']
            us_s=cr['UCL'].mean() if not cr['fixed_limits'] else cr['UCL']
            ls_s=cr['LCL'].mean() if not cr['fixed_limits'] else cr['LCL']
            nr_s=apply_nelson_rules(pt_s,us_s,cl_s,ls_s)
            lc=st.columns(3); lc[0].metric("UCL",f"{us_s:.4f}"); lc[1].metric("CL",f"{cl_s:.6f}"); lc[2].metric("LCL",f"{ls_s:.4f}")
            st.plotly_chart(plot_attribute_chart(cr,val_col,nr_s),use_container_width=True)
        st.markdown("---")
        st.markdown('<div class="sec-hdr">⚠️ Nelson Rule 이상판정 (8개)</div>',unsafe_allow_html=True)
        tot=sum(len(v) for v in nr_s['violations'].values())
        nc=st.columns(3)
        with nc[0]:
            if nr_s['any_violation']: st.error(f"🔴 이상 감지 {tot}건")
            else: st.success("🟢 모든 Rule 통과")
        nc[1].metric("Rule 1 이탈",f"{len(nr_s['violations'][1])}점")
        nc[2].metric("위반 Rule 수",f"{len([r for r,v in nr_s['violations'].items() if v])} / 8")
        rule_rows=[{"Rule":f"Rule {r}","상태":"⚠ 위반" if v else "✅ 통과",
                    "위반 부분군":str(v) if v else "-","설명":NELSON_RULE_DESC[r]}
                   for r,v in nr_s['violations'].items()]
        rdf=pd.DataFrame(rule_rows)
        def hl(row): return ["background-color:#fde8e8"]*len(row) if "위반" in row["상태"] else [""]*len(row)
        st.dataframe(rdf.style.apply(hl,axis=1),use_container_width=True,height=300)
        for msg in nr_s['summary']:
            st.markdown(f'<div class="ibox {"warn" if nr_s["any_violation"] else "ok"}">{msg}</div>',unsafe_allow_html=True)
        if show_rewrite and spc_ct in ("I-MR","Xbar-R","Xbar-S"):
            st.markdown("---")
            st.markdown('<div class="sec-hdr">🔄 이상치 제거 후 관리도 재작성</div>',unsafe_allow_html=True)
            try:
                rw=remove_outliers_and_recalc(df_raw,sg_col,val_col,spc_ct)
                if rw['removed']:
                    ini,fin=rw['initial'],rw['final']
                    if spc_ct=='I-MR': wi=ini['I_UCL']-ini['I_LCL']; wf=fin['I_UCL']-fin['I_LCL']
                    else: wi=ini['Xbar_UCL']-ini['Xbar_LCL']; wf=fin['Xbar_UCL']-fin['Xbar_LCL']
                    chg=(wf-wi)/wi*100
                    rc=st.columns(3)
                    rc[0].metric("제거 부분군",f"{len(rw['removed'])}개",help=str(rw['removed']))
                    rc[1].metric("초기 관리한계 폭",f"{wi:.4f}"); rc[2].metric("수정 관리한계 폭",f"{wf:.4f}",delta=f"{chg:.1f}%",delta_color="inverse")
                    st.plotly_chart(plot_rewrite_comparison(ini,fin),use_container_width=True)
                else: st.success("✅ 이상치 없음 — 재작성 불필요")
            except Exception as e: st.warning(str(e))
    except Exception as e:
        st.error(f"SPC 오류: {e}"); st.exception(e)

# ── TAB 4
with tab4:
    if not is_cont or cp_res is None: st.info("심층 진단은 계량형 데이터에서 제공됩니다."); st.stop()
    Cp=cp_res['Cp']; Cpk=cp_res['Cpk']; Pp=pp_res['Pp']; Ppk=pp_res['Ppk']
    Cpu=cp_res['Cpu']; Cpl=cp_res['Cpl']
    st.markdown('<div class="sec-hdr">🚦 공정 종합 판정</div>',unsafe_allow_html=True)
    icon_map={"stable":"🟢","caution":"🟡","danger":"🔴"}
    color_map={"stable":"#155724","caution":"#856404","danger":"#721c24"}
    st.markdown(f"""<div class="verdict-card {verdict_cls}">
      <div class="verdict-icon">{icon_map[verdict_cls]}</div>
      <div class="verdict-title" style="color:{color_map[verdict_cls]}">{verdict_label}</div>
      <div class="verdict-desc">{verdict_desc}</div>
    </div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📐 공정능력 지수 차이 심층 해석</div>',unsafe_allow_html=True)
    fig_wf=go.Figure(go.Waterfall(
        orientation="v",measure=["absolute","relative","relative","absolute"],
        x=["Cp\n(산포만)","중심편차\n손실","군간변동\n손실","Ppk\n(장기종합)"],
        y=[Cp,-(Cp-Cpk),-max(0,(Cp-Pp)-(Cp-Cpk)),None],
        connector=dict(line=dict(color="#bdc3c7",width=1)),
        decreasing=dict(marker_color="#e74c3c"),increasing=dict(marker_color="#27ae60"),
        totals=dict(marker_color="#4C72B0"),
        text=[f"{Cp:.4f}",f"-{Cp-Cpk:.4f}",f"-{max(0,(Cp-Pp)-(Cp-Cpk)):.4f}",f"{Ppk:.4f}"],
        textposition="outside"))
    fig_wf.update_layout(title="공정능력 손실 분해 — Cp에서 Ppk까지",
                         template="plotly_white",height=360,
                         yaxis=dict(title="지수 값",range=[0,Cp*1.3]),
                         margin=dict(l=40,r=40,t=55,b=40))
    st.plotly_chart(fig_wf,use_container_width=True)
    gap_results=interpret_index_gaps(Cp,Cpk,Pp,Ppk,Cpu,Cpl)
    for gr in gap_results:
        loss_str=f" | <b>손실률: {gr['loss_pct']:.1f}%</b>" if gr.get('loss_pct',0)>0 else ""
        st.markdown(f'<div class="ibox {gr["type"]}"><b>{gr["title"]}</b>{loss_str}<br>{gr["body"]}</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🔎 이상점 원인 후보 분석</div>',unsafe_allow_html=True)
    if nelson:
        for d in diagnose_anomaly(nelson,cp_res,pp_res):
            action_list="".join(f"<li>{a}</li>" for a in d['actions'])
            st.markdown(f"""<div class="cause-card">
              <div class="cause-title">{d['icon']} {d['cause']}</div>
              <div class="cause-body"><b>트리거:</b> {d['triggered']}<br>{d['detail']}<br><br>
              <b>권장 조치:</b><ul style="margin:4px 0 0 16px">{action_list}</ul>
              <b>관련 개념:</b> {d['related_concept']}</div>
            </div>""",unsafe_allow_html=True)

# ── TAB 5
with tab5:
    if not is_cont or cp_res is None: st.info("개선 비교는 계량형 데이터에서 제공됩니다."); st.stop()
    Cp=cp_res['Cp']; Cpk=cp_res['Cpk']
    mu=cp_res['mu']; sw=cp_res['sigma_within']
    st.markdown('<div class="sec-hdr">📐 개선 시나리오 시뮬레이터</div>',unsafe_allow_html=True)
    cur_c1,cur_c2,cur_c3=st.columns(3)
    cur_c1.metric("현재 평균(μ)",f"{mu:.4f}"); cur_c2.metric("현재 σ_within",f"{sw:.4f}"); cur_c3.metric("현재 Cpk",f"{Cpk:.4f}")
    st.markdown("---")
    imp_c1,imp_c2=st.columns(2)
    with imp_c1:
        st.markdown("##### 시나리오 A: 평균 조정 (중심 편차 제거)")
        ideal_mu=(USL+LSL)/2
        mu_after_a=st.number_input("개선 후 평균(μ')",value=round(ideal_mu,4),format="%.4f",help=f"규격 중심 = {ideal_mu:.4f}")
    with imp_c2:
        st.markdown("##### 시나리오 B: 산포 감소 (변동성 축소)")
        target_cpk=st.number_input("목표 Cpk",value=1.33,format="%.2f",min_value=0.5,max_value=3.0)
        dist_min=min(USL-mu,mu-LSL)
        sw_target=dist_min/(3*target_cpk)
        st.metric("필요한 σ",f"{sw_target:.4f}",help=f"Cpk={target_cpk:.2f} 달성을 위한 σ")
    st.markdown("---")
    res_a=simulate_improvement(mu,sw,USL,LSL,mu_after_a,sw)
    res_b=simulate_improvement(mu,sw,USL,LSL,mu,sw_target)
    col_a,col_b=st.columns(2)
    for col_,res,title in [(col_a,res_a,"시나리오 A: 평균 조정"),(col_b,res_b,"시나리오 B: 산포 감소")]:
        with col_:
            st.markdown(f"#### {title}")
            g_b=res['grade_before']; g_a=res['grade_after']
            rc1,rc2=st.columns(2)
            with rc1:
                st.markdown(f"""<div class="kpi-card {color_cls(res['before']['Cpk'])}">
                  <div class="kpi-label">개선 전 Cpk</div><div class="kpi-value">{res['before']['Cpk']:.4f}</div>
                  <span class="kpi-badge {badge_cls(res['before']['Cpk'])}">{g_b['color']} 등급{g_b['grade']}</span>
                </div>""",unsafe_allow_html=True)
            with rc2:
                dv=res['delta']['Cpk']; dc="color:#27ae60" if dv>0 else "color:#e74c3c"
                st.markdown(f"""<div class="kpi-card {color_cls(res['after']['Cpk'])}">
                  <div class="kpi-label">개선 후 Cpk</div><div class="kpi-value">{res['after']['Cpk']:.4f}</div>
                  <span class="kpi-badge {badge_cls(res['after']['Cpk'])}">{g_a['color']} 등급{g_a['grade']}</span>
                  <div class="kpi-sub"><span style="{dc}">Δ{dv:+.4f}</span></div>
                </div>""",unsafe_allow_html=True)
            st.plotly_chart(plot_improvement_dist(res,USL,LSL),use_container_width=True)
            st.plotly_chart(plot_improvement_bar(res),use_container_width=True)
            if res['delta']['Cpk']>0:
                st.markdown(f'<div class="ibox ok">✅ Cpk {res["before"]["Cpk"]:.4f} → {res["after"]["Cpk"]:.4f} (향상 <b>+{res["delta"]["Cpk"]:.4f}</b>)</div>',unsafe_allow_html=True)
            else:
                st.markdown('<div class="ibox warn">⚠ 개선 효과가 미미합니다.</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sec-hdr">📊 두 시나리오 비교</div>',unsafe_allow_html=True)
    cmp_data=pd.DataFrame({
        "항목":["현재","시나리오 A (평균조정)","시나리오 B (산포감소)"],
        "μ":[f"{mu:.4f}",f"{mu_after_a:.4f}",f"{mu:.4f}"],
        "σ":[f"{sw:.4f}",f"{sw:.4f}",f"{sw_target:.4f}"],
        "Cp":[f"{Cp:.4f}",f"{res_a['after']['Cp']:.4f}",f"{res_b['after']['Cp']:.4f}"],
        "Cpk":[f"{Cpk:.4f}",f"{res_a['after']['Cpk']:.4f}",f"{res_b['after']['Cpk']:.4f}"],
        "등급":[f"등급{get_capability_grade(Cpk)['grade']}",
                f"등급{res_a['grade_after']['grade']}",
                f"등급{res_b['grade_after']['grade']}"],
    })
    st.dataframe(cmp_data,use_container_width=True,hide_index=True)
    st.markdown("""<div class="ibox info">
    <b>핵심 개념:</b><br>
    • <b>Cp 향상</b> = 산포(σ) 감소 → 4M 변동 제어 (사람·기계·재료·방법)<br>
    • <b>Cpk 향상</b> = 중심 편차 제거 → 공정 평균을 규격 중심에 맞추기<br>
    • <b>Cp = Cpk</b>가 되면 → 중심 편차가 없는 이상적인 공정 상태
    </div>""",unsafe_allow_html=True)
