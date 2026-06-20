"""
스마트제조 공정능력분석 & 통계적공정관리 계산 함수 모음
강의록: 공정능력분석(08), 통계적공정관리(09) 기반
Author: Smart Manufacturing SPC System
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma


# ============================================================
# 0. 불편화 상수 (Unbiased Constants)
#    강의록 08_공정능력분석 7페이지 표 기반
#    SPC 강의록 09 5페이지 불편화 계수표 기반
# ============================================================

# 공정능력분석용 불편화 상수 (c4, d2, d3, d4)
# N: 부분군 크기 (2~25)
CAPABILITY_CONSTANTS = {
    #  N : (c4,       d2,     d3,     d4)
     2: (0.797885, 1.128,  0.8525, 0.954),
     3: (0.886227, 1.693,  0.8884, 1.588),
     4: (0.921318, 2.059,  0.8794, 1.978),
     5: (0.939986, 2.326,  0.8641, 2.257),
     6: (0.951533, 2.534,  0.848,  2.472),
     7: (0.959369, 2.704,  0.8332, 2.645),
     8: (0.965030, 2.847,  0.8198, 2.791),
     9: (0.969311, 2.970,  0.8078, 2.915),
    10: (0.972659, 3.078,  0.7971, 3.024),
    11: (0.975350, 3.173,  0.7873, 3.121),
    12: (0.977559, 3.258,  0.7785, 3.207),
    13: (0.979406, 3.336,  0.7704, 3.285),
    14: (0.980971, 3.407,  0.763,  3.356),
    15: (0.982316, 3.472,  0.7562, 3.422),
    16: (0.983484, 3.532,  0.7499, 3.482),
    17: (0.984506, 3.588,  0.7441, 3.538),
    18: (0.985410, 3.640,  0.7386, 3.591),
    19: (0.986214, 3.689,  0.7335, 3.640),
    20: (0.986934, 3.735,  0.7287, 3.686),
    21: (0.987583, 3.778,  0.7242, 3.730),
    22: (0.988170, 3.819,  0.7199, 3.771),
    23: (0.988705, 3.858,  0.7159, 3.811),
    24: (0.989193, 3.895,  0.7121, 3.847),
    25: (0.989640, 3.931,  0.7084, 3.883),
}

# 관리도용 불편화 상수 (A2, A3, D3, D4, B3, B4)
# 강의록 09_통계적공정관리 5페이지 표 기반
# m: 부분군 크기 (2~25)
CONTROL_CHART_CONSTANTS = {
    #  m : (A2,    A3,    d2,    D3,    D4,    B3,    B4)
     2: (1.880, 2.659, 1.128, 0.000, 3.267, 0.000, 3.267),
     3: (1.023, 1.954, 1.693, 0.000, 2.574, 0.000, 2.568),
     4: (0.729, 1.628, 2.059, 0.000, 2.282, 0.000, 2.266),
     5: (0.577, 1.427, 2.326, 0.000, 2.114, 0.000, 2.089),
     6: (0.483, 1.287, 2.534, 0.000, 2.004, 0.030, 1.970),
     7: (0.419, 1.182, 2.704, 0.076, 1.924, 0.118, 1.882),
     8: (0.373, 1.099, 2.847, 0.136, 1.864, 0.185, 1.815),
     9: (0.337, 1.032, 2.970, 0.184, 1.816, 0.239, 1.761),
    10: (0.308, 0.975, 3.078, 0.223, 1.777, 0.284, 1.716),
    11: (0.285, 0.927, 3.173, 0.256, 1.744, 0.321, 1.679),
    12: (0.266, 0.886, 3.258, 0.283, 1.717, 0.354, 1.646),
    13: (0.249, 0.850, 3.336, 0.307, 1.693, 0.382, 1.618),
    14: (0.235, 0.817, 3.407, 0.328, 1.672, 0.406, 1.594),
    15: (0.223, 0.789, 3.472, 0.347, 1.653, 0.428, 1.572),
    16: (0.212, 0.763, 3.532, 0.363, 1.637, 0.448, 1.552),
    17: (0.203, 0.739, 3.588, 0.378, 1.622, 0.466, 1.534),
    18: (0.194, 0.718, 3.640, 0.391, 1.608, 0.482, 1.518),
    19: (0.187, 0.698, 3.689, 0.403, 1.597, 0.497, 1.503),
    20: (0.180, 0.680, 3.735, 0.415, 1.585, 0.510, 1.490),
    21: (0.173, 0.663, 3.778, 0.425, 1.575, 0.523, 1.477),
    22: (0.167, 0.647, 3.819, 0.434, 1.566, 0.534, 1.466),
    23: (0.162, 0.633, 3.858, 0.443, 1.557, 0.545, 1.455),
    24: (0.157, 0.619, 3.895, 0.451, 1.548, 0.555, 1.445),
    25: (0.153, 0.606, 3.931, 0.459, 1.541, 0.565, 1.435),
}


def get_c4(n: int) -> float:
    """
    불편화 상수 c4 반환
    표준편차(s)를 모집단 표준편차(σ)로 보정할 때 사용
    강의록: 08_공정능력분석 7페이지
    """
    if 2 <= n <= 25:
        return CAPABILITY_CONSTANTS[n][0]
    else:
        # n>25 근사식: (sqrt(2)*gamma(n/2)) / (sqrt(n-1)*gamma((n-1)/2))
        return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))


def get_d2(n: int) -> float:
    """
    불편화 상수 d2 반환
    범위(R)를 σ 추정에 사용할 때 보정 계수
    강의록: 08_공정능력분석 7페이지
    """
    if 2 <= n <= 25:
        return CAPABILITY_CONSTANTS[n][1]
    else:
        # n>=51 근사식
        return 3.4873 + 0.0250141 * n - 0.00009823 * n ** 2


def get_control_coeff(name: str, m: int) -> float:
    """
    관리도용 불편화 상수 반환
    강의록: 09_통계적공정관리 5페이지 표
    
    name: 'A2', 'A3', 'd2', 'D3', 'D4', 'B3', 'B4'
    m   : 부분군 크기
    """
    col_map = {'A2': 0, 'A3': 1, 'd2': 2, 'D3': 3, 'D4': 4, 'B3': 5, 'B4': 6}
    if name not in col_map:
        raise ValueError(f"알 수 없는 상수: {name}. 가능한 값: {list(col_map.keys())}")

    if 2 <= m <= 25:
        return CONTROL_CHART_CONSTANTS[m][col_map[name]]
    else:
        # 범위 밖이면 마지막 값(m=25) 반환 (경고 출력)
        print(f"  ⚠ 경고: m={m}은 표 범위(2~25)를 벗어남. m=25 값 사용.")
        return CONTROL_CHART_CONSTANTS[25][col_map[name]]


# ============================================================
# 1. σ_within 계산 함수 (단기 표준편차)
#    강의록 08_공정능력분석 6~7페이지
# ============================================================

def calc_sigma_within(df: pd.DataFrame,
                      sg_col: str,
                      val_col: str,
                      method: str = 'pooled') -> float:
    """
    σ_within (군내 표준편차) 계산
    단기 공정능력(Cp, Cpk) 계산에 사용

    Parameters
    ----------
    df      : Long Format DataFrame
    sg_col  : 부분군 컬럼명
    val_col : 관측값 컬럼명
    method  : 계산 방법
              'pooled'  - (1) 합동표준편차(pooled std) 사용  ← 기본값
              'range'   - (2) 부분군 범위(R)의 평균 사용
              'std'     - (3) 부분군 표준편차의 평균 사용

    Returns
    -------
    sigma_within : float

    강의록 수식
    - 방법1: σ_within = sp / c4(d),  d = Σ(ni-1) + 1
    - 방법2: σ_within = R̄ / d2(ni)
    - 방법3: σ_within = s̄ / c4(ni)
    """
    groups = df.groupby(sg_col)[val_col]
    group_sizes = groups.count()
    n_mode = int(group_sizes.mode().iloc[0])  # 가장 빈도 높은 부분군 크기

    if method == 'pooled':
        # (1) 합동표준편차 방법
        # sp = sqrt( Σ Σ(xij - x̄i)² / Σ(ni-1) )
        ss_total = sum(
            np.sum((group.values - group.mean()) ** 2)
            for _, group in groups
        )
        df_total = sum(n - 1 for n in group_sizes)   # 자유도 합계
        sp = np.sqrt(ss_total / df_total)
        d = df_total + 1                              # d = Σ(ni-1) + 1
        sigma_within = sp / get_c4(d)

    elif method == 'range':
        # (2) 범위 평균 방법 (부분군 크기가 1보다 커야 함)
        R_list = [g.max() - g.min() for _, g in groups if len(g) >= 2]
        R_bar = np.mean(R_list)
        sigma_within = R_bar / get_d2(n_mode)

    elif method == 'std':
        # (3) 부분군 표준편차 평균 방법
        s_list = [g.std(ddof=1) for _, g in groups if len(g) >= 2]
        s_bar = np.mean(s_list)
        sigma_within = s_bar / get_c4(n_mode)

    else:
        raise ValueError(f"method는 'pooled', 'range', 'std' 중 하나여야 합니다.")

    return sigma_within


# ============================================================
# 2. σ_overall 계산 함수 (장기 표준편차)
#    강의록 08_공정능력분석 8페이지
# ============================================================

def calc_sigma_overall(df: pd.DataFrame,
                       sg_col: str,
                       val_col: str) -> float:
    """
    σ_overall (전체 표준편차) 계산
    장기 공정능력(Pp, Ppk) 계산에 사용

    강의록 수식:
        s = sqrt( Σ_i Σ_j (xij - x̄)² / (n-1) )
        σ_overall = s / c4(n)
        여기서 n은 전체 표본 크기

    군내변동 + 군간변동을 모두 포함한 표준편차
    """
    all_values = df[val_col].values
    n = len(all_values)
    x_bar = np.mean(all_values)

    # 전체 표준편차 (ddof=1)
    s = np.sqrt(np.sum((all_values - x_bar) ** 2) / (n - 1))

    # 불편화 상수로 보정
    sigma_overall = s / get_c4(n)

    return sigma_overall


# ============================================================
# 3. Cp, Cpk 계산 함수 (단기 공정능력)
#    강의록 08_공정능력분석 5~7페이지
# ============================================================

def calc_cp_cpk(df: pd.DataFrame,
                sg_col: str,
                val_col: str,
                USL: float,
                LSL: float,
                sigma_within_method: str = 'pooled') -> dict:
    """
    단기 공정능력지수 Cp, Cpk 계산
    σ_within (군내변동만) 사용

    강의록 수식:
        Cp  = (USL - LSL) / (6 * σ_within)          ← 산포만 반영
        Cpk = min((USL-μ)/(3σ_within), (μ-LSL)/(3σ_within))  ← 중심편차+산포

    Cp  : 공정의 순수 능력 (중심 치우침 무시)
    Cpk : 중심의 치우침과 공정의 산포 동시 고려
    Cp > Cpk 이면 → 중심 편차 존재

    Parameters
    ----------
    USL, LSL : 규격 상한/하한
    sigma_within_method : σ_within 계산 방법 ('pooled'/'range'/'std')
    """
    mu = df[val_col].mean()
    sigma_within = calc_sigma_within(df, sg_col, val_col, method=sigma_within_method)

    Cp  = (USL - LSL) / (6 * sigma_within)
    Cpu = (USL - mu)  / (3 * sigma_within)   # 상한 방향 공정능력
    Cpl = (mu  - LSL) / (3 * sigma_within)   # 하한 방향 공정능력
    Cpk = min(Cpu, Cpl)

    return {
        'Cp'           : round(Cp, 4),
        'Cpk'          : round(Cpk, 4),
        'Cpu'          : round(Cpu, 4),   # 상한 방향 (USL에 가까운 쪽)
        'Cpl'          : round(Cpl, 4),   # 하한 방향 (LSL에 가까운 쪽)
        'mu'           : round(mu, 4),
        'sigma_within' : round(sigma_within, 4),
        'USL'          : USL,
        'LSL'          : LSL,
    }


# ============================================================
# 4. Pp, Ppk 계산 함수 (장기 공정능력)
#    강의록 08_공정능력분석 8페이지
# ============================================================

def calc_pp_ppk(df: pd.DataFrame,
                sg_col: str,
                val_col: str,
                USL: float,
                LSL: float) -> dict:
    """
    장기 공정능력지수 Pp, Ppk 계산
    σ_overall (군내+군간변동 모두) 사용

    강의록 수식:
        Pp  = (USL - LSL) / (6 * σ_overall)
        Ppk = min((USL-μ)/(3σ_overall), (μ-LSL)/(3σ_overall))

    Cp vs Pp 갭이 크면 → 군간변동(between variance)이 크다는 의미
    → 로트·교대·시간 등에 따른 변동 존재
    """
    mu = df[val_col].mean()
    sigma_overall = calc_sigma_overall(df, sg_col, val_col)

    Pp  = (USL - LSL) / (6 * sigma_overall)
    Ppu = (USL - mu)  / (3 * sigma_overall)
    Ppl = (mu  - LSL) / (3 * sigma_overall)
    Ppk = min(Ppu, Ppl)

    return {
        'Pp'            : round(Pp, 4),
        'Ppk'           : round(Ppk, 4),
        'Ppu'           : round(Ppu, 4),
        'Ppl'           : round(Ppl, 4),
        'mu'            : round(mu, 4),
        'sigma_overall' : round(sigma_overall, 4),
        'USL'           : USL,
        'LSL'           : LSL,
    }


# ============================================================
# 5. 공정능력 등급 판정
#    강의록 08_공정능력분석 9페이지 판정기준표
# ============================================================

def get_capability_grade(Cp: float) -> dict:
    """
    공정능력 등급 판정 (강의록 판정기준표 기반)

    등급 0: Cp >= 1.67  → 공정능력 매우 충분  (±5σ)
    등급 1: Cp >= 1.33  → 공정능력 충분       (±4σ)
    등급 2: Cp >= 1.00  → 충분하지 않으나 괜찮다 (±3σ)
    등급 3: Cp >= 0.67  → 공정능력 모자라다   (±2σ)
    등급 4: Cp <  0.67  → 공정능력 매우 부족  (±1σ)
    """
    if Cp >= 1.67:
        return {
            'grade': 0, 'status': '매우 충분',
            'sigma_level': '±5σ', 'color': '🟢',
            'action': '들쭉날쭉이 약간 커져도 걱정 없음. 비용절감·관리 간소화 검토 가능'
        }
    elif Cp >= 1.33:
        return {
            'grade': 1, 'status': '충분',
            'sigma_level': '±4σ', 'color': '🟢',
            'action': '아주 이상적인 공정상황. 현재 상태 유지'
        }
    elif Cp >= 1.00:
        return {
            'grade': 2, 'status': '충분하지 않으나 그 정도면 괜찮다',
            'sigma_level': '±3σ', 'color': '🟡',
            'action': '공정 관리를 확실히 하여 관리상태 유지. Cp가 1에 가까우면 불량 발생 가능'
        }
    elif Cp >= 0.67:
        return {
            'grade': 3, 'status': '모자라다',
            'sigma_level': '±2σ', 'color': '🔴',
            'action': '불량품 발생 중. 전체 선별 및 공정 개선·관리 필요'
        }
    else:
        return {
            'grade': 4, 'status': '매우 부족',
            'sigma_level': '±1σ', 'color': '🔴',
            'action': '품질이 전혀 만족스럽지 않음. 긴급 대책 필요. 규격 재검토 고려'
        }


# ============================================================
# 6. 관리도 자동 선택 함수
#    강의록 09_통계적공정관리 2~3페이지 flowchart 기반
# ============================================================

def select_control_chart(data_type: str,
                         subgroup_size: int,
                         size_varies: bool = False,
                         attribute_type: str = None) -> dict:
    """
    강의록 관리도 선택 기준 flowchart를 코드로 구현

    Parameters
    ----------
    data_type      : 'continuous' (계량형) or 'attribute' (계수형)
    subgroup_size  : 부분군 크기 (최빈값 기준)
    size_varies    : 부분군 크기 변동 여부 (True이면 크기가 일정하지 않음)
    attribute_type : 계수형일 때 'defective' (불량) or 'defect' (결함)

    Returns
    -------
    dict: {chart_type, reason, distribution}

    강의록 선택 기준
    - 계량형 + n=1       → I-MR  (부분군 크기 1, 이동범위)
    - 계량형 + 2≤n≤9    → Xbar-R (평균-범위)
    - 계량형 + n≥10     → Xbar-S (평균-표준편차)
    - 계수형 + 불량 + 일정 크기 → NP (불량개수)
    - 계수형 + 불량 + 가변 크기 → P  (불량률)
    - 계수형 + 결함 + 일정 크기 → C  (결함수)
    - 계수형 + 결함 + 가변 크기 → U  (단위당 결함수)
    """
    if data_type == 'continuous':
        if subgroup_size == 1:
            return {
                'chart_type'  : 'I-MR',
                'reason'      : f'계량형 데이터, 부분군 크기 n=1 → I-MR 관리도 사용',
                'distribution': '정규분포',
                'charts'      : ['I chart', 'MR chart'],
            }
        elif 2 <= subgroup_size <= 9:
            return {
                'chart_type'  : 'Xbar-R',
                'reason'      : f'계량형 데이터, 부분군 크기 2≤n={subgroup_size}≤9 → Xbar-R 관리도 사용',
                'distribution': '정규분포',
                'charts'      : ['Xbar chart', 'R chart'],
            }
        else:  # subgroup_size >= 10
            return {
                'chart_type'  : 'Xbar-S',
                'reason'      : f'계량형 데이터, 부분군 크기 n={subgroup_size}≥10 → Xbar-S 관리도 사용',
                'distribution': '정규분포',
                'charts'      : ['Xbar chart', 'S chart'],
            }

    elif data_type == 'attribute':
        if attribute_type == 'defective':   # 불량 (이항분포)
            if not size_varies:
                return {
                    'chart_type'  : 'NP',
                    'reason'      : '계수형(불량), 부분군 크기 일정 → NP 관리도 (불량개수)',
                    'distribution': '이항분포(Binomial)',
                    'charts'      : ['NP chart'],
                }
            else:
                return {
                    'chart_type'  : 'P',
                    'reason'      : '계수형(불량), 부분군 크기 가변 → P 관리도 (불량률)',
                    'distribution': '이항분포(Binomial)',
                    'charts'      : ['P chart'],
                }

        elif attribute_type == 'defect':    # 결함 (포아송분포)
            if not size_varies:
                return {
                    'chart_type'  : 'C',
                    'reason'      : '계수형(결함), 부분군 크기 일정 → C 관리도 (결함수)',
                    'distribution': '포아송분포(Poisson)',
                    'charts'      : ['C chart'],
                }
            else:
                return {
                    'chart_type'  : 'U',
                    'reason'      : '계수형(결함), 부분군 크기 가변 → U 관리도 (단위당 결함수)',
                    'distribution': '포아송분포(Poisson)',
                    'charts'      : ['U chart'],
                }
        else:
            raise ValueError("attribute_type은 'defective' 또는 'defect'여야 합니다.")

    else:
        raise ValueError("data_type은 'continuous' 또는 'attribute'여야 합니다.")


# ============================================================
# 7. I-MR 관리도 계산 함수
#    강의록 09_통계적공정관리 5페이지 (부분군 크기 n=1)
# ============================================================

def calc_imr_chart(df: pd.DataFrame,
                   sg_col: str,
                   val_col: str,
                   window: int = 3) -> dict:
    """
    I-MR (개별값-이동범위) 관리도 계산
    부분군 크기가 1인 경우 사용

    강의록 수식:
        I chart:
            CL  = X̄
            UCL = X̄ + 3 * MR̄ / d2(w)  ← E2(w) = 3/d2(w)
            LCL = X̄ - 3 * MR̄ / d2(w)

        MR chart:
            CL  = MR̄
            UCL = D4(w) * MR̄
            LCL = D3(w) * MR̄ = 0  (w<=2이면 항상 0)

    Parameters
    ----------
    window : 이동범위 계산 윈도우 크기 w (기본값=3, 강의록 예제 기준)
    """
    # 부분군별 정렬 후 개별값 추출
    data = df.sort_values(sg_col)[val_col].values
    n = len(data)

    # 이동범위(MR_i) 계산
    # w=3이면: MR_i = max(x_i, x_{i+1}, x_{i+2}) - min(x_i, x_{i+1}, x_{i+2})
    MR_list = []
    for i in range(n - window + 1):
        segment = data[i: i + window]
        MR_list.append(segment.max() - segment.min())
    MR_values = np.array(MR_list)

    Xbar   = np.mean(data)
    MR_bar = np.mean(MR_values)

    # 불편화 상수
    d2_w = get_d2(window)
    D3_w = get_control_coeff('D3', window)
    D4_w = get_control_coeff('D4', window)

    # I chart 관리한계
    I_UCL = Xbar + 3 * MR_bar / d2_w
    I_CL  = Xbar
    I_LCL = Xbar - 3 * MR_bar / d2_w

    # MR chart 관리한계
    MR_UCL = D4_w * MR_bar
    MR_CL  = MR_bar
    MR_LCL = D3_w * MR_bar    # 대부분 0

    # 관측치 인덱스 정렬
    sg_values  = df.sort_values(sg_col)[sg_col].unique()
    mr_indices = sg_values[window - 1:]  # MR은 window-1번째부터 시작

    return {
        'chart_type': 'I-MR',
        'window'    : window,
        # I chart
        'I_point'   : pd.Series(data, index=sg_values),
        'I_UCL'     : round(I_UCL, 4),
        'I_CL'      : round(I_CL,  4),
        'I_LCL'     : round(I_LCL, 4),
        # MR chart
        'MR_point'  : pd.Series(MR_values, index=mr_indices),
        'MR_UCL'    : round(MR_UCL, 4),
        'MR_CL'     : round(MR_CL,  4),
        'MR_LCL'    : round(MR_LCL, 4),
        # 통계량
        'Xbar'      : round(Xbar, 4),
        'MR_bar'    : round(MR_bar, 4),
        'd2'        : d2_w,
        'D3'        : D3_w,
        'D4'        : D4_w,
    }


# ============================================================
# 8. Xbar-R 관리도 계산 함수
#    강의록 09_통계적공정관리 5페이지 (부분군 크기 2~9)
# ============================================================

def calc_xbar_r_chart(df: pd.DataFrame,
                      sg_col: str,
                      val_col: str) -> dict:
    """
    Xbar-R (평균-범위) 관리도 계산
    부분군 크기 2 ≤ n ≤ 9 인 경우 사용

    강의록 수식:
        Xbar chart:
            CL  = X̄̄  (부분군 평균의 평균)
            UCL = X̄̄ + A2 * R̄
            LCL = X̄̄ - A2 * R̄

        R chart:
            CL  = R̄
            UCL = D4 * R̄
            LCL = D3 * R̄  (n≤6이면 0)
    """
    groups = df.groupby(sg_col)[val_col]

    # 부분군별 통계량 계산
    sg_stats = pd.DataFrame({
        'Xbar'  : groups.mean(),
        'R'     : groups.max() - groups.min(),
        'n_i'   : groups.count(),
    })

    # 최빈 부분군 크기
    m = int(sg_stats['n_i'].mode().iloc[0])

    # 불편화 상수 (최빈 크기 기준)
    A2 = get_control_coeff('A2', m)
    D3 = get_control_coeff('D3', m)
    D4 = get_control_coeff('D4', m)

    Xbar_bar = sg_stats['Xbar'].mean()   # X double bar
    R_bar    = sg_stats['R'].mean()

    # Xbar chart 관리한계
    Xbar_UCL = Xbar_bar + A2 * R_bar
    Xbar_CL  = Xbar_bar
    Xbar_LCL = Xbar_bar - A2 * R_bar

    # R chart 관리한계
    R_UCL = D4 * R_bar
    R_CL  = R_bar
    R_LCL = D3 * R_bar

    return {
        'chart_type' : 'Xbar-R',
        'n'          : m,
        # Xbar chart
        'Xbar_point' : sg_stats['Xbar'],
        'Xbar_UCL'   : round(Xbar_UCL, 4),
        'Xbar_CL'    : round(Xbar_CL,  4),
        'Xbar_LCL'   : round(Xbar_LCL, 4),
        # R chart
        'R_point'    : sg_stats['R'],
        'R_UCL'      : round(R_UCL, 4),
        'R_CL'       : round(R_CL,  4),
        'R_LCL'      : round(R_LCL, 4),
        # 통계량
        'Xbar_bar'   : round(Xbar_bar, 4),
        'R_bar'      : round(R_bar, 4),
        'A2'         : A2,
        'D3'         : D3,
        'D4'         : D4,
    }


# ============================================================
# 9. Xbar-S 관리도 계산 함수
#    강의록 09_통계적공정관리 5페이지 (부분군 크기 ≥ 10)
# ============================================================

def calc_xbar_s_chart(df: pd.DataFrame,
                      sg_col: str,
                      val_col: str) -> dict:
    """
    Xbar-S (평균-표준편차) 관리도 계산
    부분군 크기 n ≥ 10 인 경우 사용

    강의록 수식:
        Xbar chart:
            CL  = X̄̄
            UCL = X̄̄ + A3 * s̄
            LCL = X̄̄ - A3 * s̄

        S chart:
            CL  = s̄
            UCL = B4 * s̄
            LCL = B3 * s̄  (n≤5이면 0, 실제 n≥10 사용이므로 양수)
    """
    groups = df.groupby(sg_col)[val_col]

    sg_stats = pd.DataFrame({
        'Xbar' : groups.mean(),
        's'    : groups.std(ddof=1),
        'n_i'  : groups.count(),
    })

    m = int(sg_stats['n_i'].mode().iloc[0])

    A3 = get_control_coeff('A3', m)
    B3 = get_control_coeff('B3', m)
    B4 = get_control_coeff('B4', m)

    Xbar_bar = sg_stats['Xbar'].mean()
    s_bar    = sg_stats['s'].mean()

    Xbar_UCL = Xbar_bar + A3 * s_bar
    Xbar_CL  = Xbar_bar
    Xbar_LCL = Xbar_bar - A3 * s_bar

    S_UCL = B4 * s_bar
    S_CL  = s_bar
    S_LCL = B3 * s_bar

    return {
        'chart_type' : 'Xbar-S',
        'n'          : m,
        # Xbar chart
        'Xbar_point' : sg_stats['Xbar'],
        'Xbar_UCL'   : round(Xbar_UCL, 4),
        'Xbar_CL'    : round(Xbar_CL,  4),
        'Xbar_LCL'   : round(Xbar_LCL, 4),
        # S chart
        'S_point'    : sg_stats['s'],
        'S_UCL'      : round(S_UCL, 4),
        'S_CL'       : round(S_CL,  4),
        'S_LCL'      : round(S_LCL, 4),
        # 통계량
        'Xbar_bar'   : round(Xbar_bar, 4),
        's_bar'      : round(s_bar, 4),
        'A3'         : A3,
        'B3'         : B3,
        'B4'         : B4,
    }


# ============================================================
# 10. 계수형 관리도 계산 함수 (P, NP, C, U)
#     강의록 09_통계적공정관리 5페이지
# ============================================================

def calc_attribute_chart(df: pd.DataFrame,
                         sg_col: str,
                         n_col: str,
                         val_col: str,
                         chart_type: str) -> dict:
    """
    계수형 관리도 (NP, P, C, U) 계산

    Parameters
    ----------
    df         : DataFrame (sg_col, n_col, val_col 포함)
    sg_col     : 부분군(로트) 컬럼명
    n_col      : 표본 크기 컬럼명 (부분군별 검사 개수)
    val_col    : 불량수 또는 결함수 컬럼명
    chart_type : 'NP', 'P', 'C', 'U'

    강의록 수식
    ──────────────────────────────────────────
    NP (불량개수, 부분군 크기 일정):
        np̄ = Σnp / k  (k: 부분군 수)
        p̄  = Σnp / Σn
        UCL = np̄ + 3√(np̄(1-p̄))
        LCL = np̄ - 3√(np̄(1-p̄))

    P (불량률, 부분군 크기 가변):
        p̄  = Σnp / Σn
        UCL_i = p̄ + 3√(p̄(1-p̄)/n_i)
        LCL_i = p̄ - 3√(p̄(1-p̄)/n_i)

    C (결함수, 부분군 크기 일정):
        c̄  = Σc / k
        UCL = c̄ + 3√c̄
        LCL = c̄ - 3√c̄

    U (단위당 결함수, 부분군 크기 가변):
        ū   = Σc / Σn
        UCL_i = ū + 3√(ū/n_i)
        LCL_i = ū - 3√(ū/n_i)
    """
    data = df.set_index(sg_col).copy()
    n_i  = data[n_col].values       # 표본 크기 배열
    obs  = data[val_col].values      # 관측값(불량수/결함수) 배열
    k    = len(obs)                  # 부분군 수

    if chart_type == 'NP':
        np_bar = obs.sum() / k            # 평균 불량개수
        p_bar  = obs.sum() / n_i.sum()    # 평균 불량률

        UCL = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        CL  = np_bar
        LCL = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))

        point = pd.Series(obs, index=data.index)
        return {
            'chart_type': 'NP', 'point': point,
            'UCL': round(UCL, 4), 'CL': round(CL, 4), 'LCL': round(LCL, 4),
            'np_bar': round(np_bar, 4), 'p_bar': round(p_bar, 6),
            'fixed_limits': True,   # 한계선이 고정 (일정 크기)
        }

    elif chart_type == 'P':
        p_bar = obs.sum() / n_i.sum()     # 평균 불량률

        point  = obs / n_i                # 부분군별 불량률
        UCL_i  = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n_i)
        LCL_i  = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n_i))

        return {
            'chart_type': 'P', 'point': pd.Series(point, index=data.index),
            'UCL': pd.Series(UCL_i, index=data.index),
            'CL' : round(p_bar, 6),
            'LCL': pd.Series(LCL_i, index=data.index),
            'p_bar': round(p_bar, 6),
            'fixed_limits': False,  # 한계선이 부분군별로 다름
        }

    elif chart_type == 'C':
        c_bar = obs.mean()                # 평균 결함수

        UCL = c_bar + 3 * np.sqrt(c_bar)
        CL  = c_bar
        LCL = max(0, c_bar - 3 * np.sqrt(c_bar))

        return {
            'chart_type': 'C', 'point': pd.Series(obs, index=data.index),
            'UCL': round(UCL, 4), 'CL': round(CL, 4), 'LCL': round(LCL, 4),
            'c_bar': round(c_bar, 4),
            'fixed_limits': True,
        }

    elif chart_type == 'U':
        u_bar = obs.sum() / n_i.sum()     # 단위당 평균 결함수

        point  = obs / n_i
        UCL_i  = u_bar + 3 * np.sqrt(u_bar / n_i)
        LCL_i  = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / n_i))

        return {
            'chart_type': 'U', 'point': pd.Series(point, index=data.index),
            'UCL': pd.Series(UCL_i, index=data.index),
            'CL' : round(u_bar, 6),
            'LCL': pd.Series(LCL_i, index=data.index),
            'u_bar': round(u_bar, 6),
            'fixed_limits': False,
        }

    else:
        raise ValueError(f"지원하지 않는 chart_type: {chart_type}. 'NP','P','C','U' 중 선택")


# ============================================================
# 11. Nelson Rule 이상 판정 함수 (8개 Rule)
#     강의록 09_통계적공정관리 4페이지
# ============================================================

NELSON_RULE_DESC = {
    1: "Rule 1: 1개 이상의 관측치가 ±3σ를 벗어난 경우 (관리한계 이탈)",
    2: "Rule 2: 연속 9점이 CL 위 또는 아래에 존재 (편향, Bias 존재)",
    3: "Rule 3: 연속 6점이 단조 증가 또는 감소 (추세, Trend 존재)",
    4: "Rule 4: 연속 14점이 교대로 증가/감소 (진동, Oscillation 존재)",
    5: "Rule 5: 연속 3점 중 2점이 ±2σ를 같은 방향으로 이탈",
    6: "Rule 6: 연속 5점 중 4점이 ±1σ를 같은 방향으로 이탈",
    7: "Rule 7: 연속 15점이 모두 ±1σ 범위 이내 (층화, Stratification)",
    8: "Rule 8: 연속 8점이 모두 ±1σ 밖에 존재 (혼합, Mixture)",
}


def apply_nelson_rules(point: pd.Series,
                       UCL: float,
                       CL: float,
                       LCL: float) -> dict:
    """
    Nelson Rule 8개 적용하여 이상점 탐지
    강의록 09_통계적공정관리 4페이지 기준

    Parameters
    ----------
    point : 관측값 Series (인덱스는 부분군 번호)
    UCL   : 관리 상한선
    CL    : 중심선
    LCL   : 관리 하한선

    Returns
    -------
    dict:
        'violations': {rule번호: [위반 인덱스]}
        'any_violation': bool
        'summary': 설명 문자열 리스트
    """
    data   = point.values
    idx    = point.index.tolist()
    n      = len(data)
    sigma  = (UCL - CL) / 3     # 1σ 크기

    violations = {i: [] for i in range(1, 9)}

    for i in range(n):
        val = data[i]

        # ── Rule 1: ±3σ 이탈 ──────────────────────────────
        if val > UCL or val < LCL:
            violations[1].append(idx[i])

        # ── Rule 2: 연속 9점이 CL 한쪽에 존재 ─────────────
        if i >= 8:
            w = data[i - 8: i + 1]
            if all(x > CL for x in w) or all(x < CL for x in w):
                violations[2].append(idx[i])

        # ── Rule 3: 연속 6점 단조 증가/감소 ────────────────
        if i >= 5:
            w = data[i - 5: i + 1]
            increasing = all(w[j] < w[j + 1] for j in range(5))
            decreasing = all(w[j] > w[j + 1] for j in range(5))
            if increasing or decreasing:
                violations[3].append(idx[i])

        # ── Rule 4: 연속 14점 교대 증가/감소 ───────────────
        if i >= 13:
            w = data[i - 13: i + 1]
            alternating = all(
                (w[j] < w[j + 1]) != (w[j + 1] < w[j + 2])
                for j in range(12)
            )
            if alternating:
                violations[4].append(idx[i])

        # ── Rule 5: 연속 3점 중 2점이 ±2σ 이탈 (동일 방향)
        if i >= 2:
            w = data[i - 2: i + 1]
            above2 = sum(1 for x in w if x > CL + 2 * sigma)
            below2 = sum(1 for x in w if x < CL - 2 * sigma)
            if above2 >= 2 or below2 >= 2:
                violations[5].append(idx[i])

        # ── Rule 6: 연속 5점 중 4점이 ±1σ 이탈 (동일 방향)
        if i >= 4:
            w = data[i - 4: i + 1]
            above1 = sum(1 for x in w if x > CL + sigma)
            below1 = sum(1 for x in w if x < CL - sigma)
            if above1 >= 4 or below1 >= 4:
                violations[6].append(idx[i])

        # ── Rule 7: 연속 15점이 ±1σ 이내 (층화) ───────────
        if i >= 14:
            w = data[i - 14: i + 1]
            if all(CL - sigma < x < CL + sigma for x in w):
                violations[7].append(idx[i])

        # ── Rule 8: 연속 8점이 ±1σ 밖 양쪽 (혼합) ─────────
        if i >= 7:
            w = data[i - 7: i + 1]
            if all(x > CL + sigma or x < CL - sigma for x in w):
                violations[8].append(idx[i])

    # 요약 생성
    any_violation = any(len(v) > 0 for v in violations.values())
    summary = []
    for rule, viol_idx in violations.items():
        if viol_idx:
            summary.append(
                f"  ⚠ {NELSON_RULE_DESC[rule]}\n"
                f"    위반 부분군: {viol_idx}"
            )

    if not any_violation:
        summary.append("  ✅ 모든 Nelson Rule을 통과했습니다. 공정이 관리 상태입니다.")

    return {
        'violations'   : violations,
        'any_violation': any_violation,
        'summary'      : summary,
    }


# ============================================================
# 12. 이상치 제거 후 관리도 재작성 함수
#     강의록 09_통계적공정관리 26~28페이지
# ============================================================

def remove_outliers_and_recalc(df: pd.DataFrame,
                                sg_col: str,
                                val_col: str,
                                chart_type: str,
                                max_iter: int = 5) -> dict:
    """
    이상치 제거 후 관리도 재작성 (반복 절차)
    강의록: 이상점 발견 → 이상원인 제거 → 재작성 → 반복

    절차 (강의록 기준):
        1단계: 관리도 작성 → 이상점(UCL/LCL 이탈) 확인
        2단계: 이상 부분군 제거 → 남은 데이터로 관리도 재작성
        3단계: 모든 점이 관리상태가 될 때까지 반복
        4단계: 마지막 단계 관리한계를 채택하여 공정에 적용

    Parameters
    ----------
    max_iter : 최대 반복 횟수 (무한루프 방지)

    Returns
    -------
    dict: {
        'initial': 초기 관리도 결과,
        'final'  : 최종 관리도 결과,
        'removed': 제거된 부분군 목록,
        'iterations': 반복 이력
    }
    """
    calc_func = {
        'Xbar-R': calc_xbar_r_chart,
        'Xbar-S': calc_xbar_s_chart,
        'I-MR'  : calc_imr_chart,
    }
    if chart_type not in calc_func:
        raise ValueError(f"현재 재작성은 계량형 관리도만 지원: {list(calc_func.keys())}")

    df_current  = df.copy()
    removed_all = []
    iterations  = []

    # 1단계: 초기 관리도
    initial = calc_func[chart_type](df_current, sg_col, val_col)
    iterations.append({'step': 0, 'result': initial, 'removed': []})

    for step in range(1, max_iter + 1):
        chart = calc_func[chart_type](df_current, sg_col, val_col)

        # Xbar chart 기준으로 이상점 탐지 (Rule 1만 적용: ±3σ 이탈)
        if chart_type == 'I-MR':
            point_series = chart['I_point']
            UCL, LCL     = chart['I_UCL'], chart['I_LCL']
        else:
            point_series = chart['Xbar_point']
            UCL, LCL     = chart['Xbar_UCL'], chart['Xbar_LCL']

        ooc = point_series[
            (point_series > UCL) | (point_series < LCL)
        ].index.tolist()

        if not ooc:
            break   # 이상점 없으면 종료

        # 이상 부분군 제거
        df_current  = df_current[~df_current[sg_col].isin(ooc)].copy()
        removed_all.extend(ooc)
        iterations.append({'step': step, 'result': chart, 'removed': ooc})

    final = calc_func[chart_type](df_current, sg_col, val_col)

    return {
        'initial'   : initial,
        'final'     : final,
        'removed'   : removed_all,
        'iterations': iterations,
        'n_initial' : len(df),
        'n_final'   : len(df_current),
    }


# ============================================================
# 13. 공정능력 자동 해석 함수
#     강의록 08_공정능력분석 판정기준표 기반
# ============================================================

def interpret_capability(cp_result: dict, pp_result: dict) -> list:
    """
    Cp, Cpk, Pp, Ppk 결과를 받아 강의록 기준으로 자동 해석 문장 생성

    강의록 해석 기준:
    ① Cp 등급 판정 → 기본 공정능력 수준
    ② Cp vs Cpk 갭 → 중심 편차 여부
    ③ Cp vs Pp 갭 → 군간변동 크기
    ④ Cpk 방향성 → USL/LSL 중 어느 쪽 위험
    """
    Cp  = cp_result['Cp']
    Cpk = cp_result['Cpk']
    Cpu = cp_result['Cpu']
    Cpl = cp_result['Cpl']
    Pp  = pp_result['Pp']
    Ppk = pp_result['Ppk']

    messages = []

    # ① Cp 등급 판정
    grade_info = get_capability_grade(Cp)
    messages.append(
        f"{grade_info['color']} [공정능력 등급 {grade_info['grade']}] "
        f"Cp = {Cp:.4f} → {grade_info['status']} ({grade_info['sigma_level']})\n"
        f"   권장 조치: {grade_info['action']}"
    )

    # ② Cp vs Cpk: 중심 편차 해석
    gap_cp_cpk = Cp - Cpk
    if gap_cp_cpk > 0.1:
        messages.append(
            f"📍 중심 편차 존재: Cp({Cp:.4f}) > Cpk({Cpk:.4f}), 차이 = {gap_cp_cpk:.4f}\n"
            f"   공정 평균이 규격 중심에서 벗어나 있습니다. 공정 평균을 조정하세요."
        )
    else:
        messages.append(
            f"✅ 중심 정렬 양호: Cp ≈ Cpk (차이 = {gap_cp_cpk:.4f})\n"
            f"   공정 평균이 규격 중심과 잘 일치합니다."
        )

    # ③ Cp vs Pp: 군간변동 해석
    gap_cp_pp = Cp - Pp
    if gap_cp_pp > 0.1:
        messages.append(
            f"📍 군간변동 큼: Cp({Cp:.4f}) > Pp({Pp:.4f}), 차이 = {gap_cp_pp:.4f}\n"
            f"   로트·교대근무·시간에 따른 공정 변동이 존재합니다. 군간 원인을 조사하세요."
        )
    else:
        messages.append(
            f"✅ 군간변동 안정: Cp ≈ Pp (차이 = {gap_cp_pp:.4f})\n"
            f"   군간변동이 크지 않습니다. 공정이 장기적으로도 안정적입니다."
        )

    # ④ Cpk 방향성: 어느 쪽에 불량 위험이 큰가
    if Cpu < Cpl:
        messages.append(
            f"⬆ USL 방향 불량 위험: Cpu({Cpu:.4f}) < Cpl({Cpl:.4f})\n"
            f"   공정 평균이 USL(규격 상한) 방향으로 치우쳐 있습니다."
        )
    elif Cpl < Cpu:
        messages.append(
            f"⬇ LSL 방향 불량 위험: Cpl({Cpl:.4f}) < Cpu({Cpu:.4f})\n"
            f"   공정 평균이 LSL(규격 하한) 방향으로 치우쳐 있습니다."
        )
    else:
        messages.append("⚖ 양쪽 방향 위험이 동일합니다.")

    return messages


# ============================================================
# ✅ 테스트 코드
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  스마트제조 SPC 계산 함수 테스트")
    print("  강의록 PVC 점도 예제 데이터 사용")
    print("=" * 65)

    # ── 강의록 08 예제 데이터: PVC 점도 (6개 생산라인, 각 5개 샘플) ──
    raw = np.array([
        [3576.27, 3630.12, 3576.27, 3630.12, 3355.69, 3363.62],
        [3504.17, 3514.52, 3747.43, 3666.15, 3709.25, 3317.28],
        [3440.11, 3494.35, 3962.93, 3514.30, 3273.57, 3336.20],
        [3638.33, 3719.84, 3617.47, 3450.17, 3378.70, 3475.50],
        [3661.94, 3485.53, 3499.43, 3605.53, 3390.29, 3519.26],
    ])
    df_wide = pd.DataFrame(raw, columns=[f'pl_{i}' for i in range(1, 7)])
    df_long = df_wide.melt(var_name='prod_line', value_name='viscocity')

    USL_PVC = 3500 + 500   # 4000
    LSL_PVC = 3500 - 500   # 3000

    print("\n▶ 데이터 미리보기 (head)")
    print(df_long.head(8).to_string(index=False))

    # ── 1. σ_within 계산 ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[1] σ_within 계산 (3가지 방법)")
    for method in ['pooled', 'range', 'std']:
        sw = calc_sigma_within(df_long, 'prod_line', 'viscocity', method=method)
        print(f"  방법 '{method}': σ_within = {sw:.4f}")

    # ── 2. σ_overall 계산 ────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[2] σ_overall 계산")
    so = calc_sigma_overall(df_long, 'prod_line', 'viscocity')
    print(f"  σ_overall = {so:.4f}")

    # ── 3. Cp, Cpk 계산 ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[3] Cp, Cpk 계산 (단기 공정능력)")
    cp_res = calc_cp_cpk(df_long, 'prod_line', 'viscocity', USL_PVC, LSL_PVC)
    for k, v in cp_res.items():
        print(f"  {k:15s} = {v}")

    # ── 4. Pp, Ppk 계산 ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[4] Pp, Ppk 계산 (장기 공정능력)")
    pp_res = calc_pp_ppk(df_long, 'prod_line', 'viscocity', USL_PVC, LSL_PVC)
    for k, v in pp_res.items():
        print(f"  {k:15s} = {v}")

    # ── 5. 공정능력 등급 판정 ─────────────────────────────────────
    print("\n" + "─" * 50)
    print("[5] 공정능력 등급 판정")
    grade = get_capability_grade(cp_res['Cp'])
    print(f"  {grade['color']} 등급 {grade['grade']}: {grade['status']} ({grade['sigma_level']})")
    print(f"  권장 조치: {grade['action']}")

    # ── 6. 관리도 자동 선택 ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("[6] 관리도 자동 선택")
    for n_test in [1, 5, 10]:
        sel = select_control_chart('continuous', n_test)
        print(f"  n={n_test:2d} → {sel['chart_type']:7s} | {sel['reason']}")

    sel_p  = select_control_chart('attribute', 200, size_varies=False, attribute_type='defective')
    sel_u  = select_control_chart('attribute', 200, size_varies=True,  attribute_type='defect')
    print(f"  불량+일정 → {sel_p['chart_type']} | {sel_p['reason']}")
    print(f"  결함+가변 → {sel_u['chart_type']} | {sel_u['reason']}")

    # ── 7. Xbar-R 관리도 계산 ────────────────────────────────────
    print("\n" + "─" * 50)
    print("[7] Xbar-R 관리도 계산 (부분군 크기=5)")
    xr = calc_xbar_r_chart(df_long, 'prod_line', 'viscocity')
    print(f"  Xbar chart: UCL={xr['Xbar_UCL']}, CL={xr['Xbar_CL']}, LCL={xr['Xbar_LCL']}")
    print(f"  R chart:    UCL={xr['R_UCL']},    CL={xr['R_CL']},    LCL={xr['R_LCL']}")
    print(f"  (A2={xr['A2']}, D3={xr['D3']}, D4={xr['D4']}, n={xr['n']})")

    # ── 8. I-MR 관리도 계산 ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("[8] I-MR 관리도 계산 (개별값 데이터, window=3)")
    # 개별 데이터 생성 (부분군 크기=1)
    np.random.seed(42)
    df_imr = pd.DataFrame({
        'lot'      : range(1, 21),
        'thickness': np.random.normal(40, 2, 20),
    })
    imr = calc_imr_chart(df_imr, 'lot', 'thickness', window=3)
    print(f"  I  chart: UCL={imr['I_UCL']}, CL={imr['I_CL']}, LCL={imr['I_LCL']}")
    print(f"  MR chart: UCL={imr['MR_UCL']}, CL={imr['MR_CL']}, LCL={imr['MR_LCL']}")

    # ── 9. Xbar-S 관리도 계산 ────────────────────────────────────
    print("\n" + "─" * 50)
    print("[9] Xbar-S 관리도 계산 (부분군 크기=10)")
    np.random.seed(0)
    rows = []
    for sg in range(1, 21):
        for _ in range(10):
            rows.append({'subgroup': sg, 'value': np.random.normal(100, 5)})
    df_xs = pd.DataFrame(rows)
    xs = calc_xbar_s_chart(df_xs, 'subgroup', 'value')
    print(f"  Xbar chart: UCL={xs['Xbar_UCL']}, CL={xs['Xbar_CL']}, LCL={xs['Xbar_LCL']}")
    print(f"  S chart:    UCL={xs['S_UCL']},    CL={xs['S_CL']},    LCL={xs['S_LCL']}")
    print(f"  (A3={xs['A3']}, B3={xs['B3']}, B4={xs['B4']}, n={xs['n']})")

    # ── 10. 계수형 관리도 (NP, P, C, U) ─────────────────────────
    print("\n" + "─" * 50)
    print("[10] 계수형 관리도 계산")
    np.random.seed(7)
    n_lots = 20
    df_attr = pd.DataFrame({
        'Lot'        : range(1, n_lots + 1),
        'sample_size': np.random.randint(190, 215, n_lots),
        'Defectives' : np.random.binomial(200, 0.02, n_lots),
    })

    for ctype in ['NP', 'P']:
        res = calc_attribute_chart(df_attr, 'Lot', 'sample_size', 'Defectives', ctype)
        if res['fixed_limits']:
            print(f"  {ctype}: UCL={res['UCL']}, CL={res['CL']}, LCL={res['LCL']}")
        else:
            ucl_val = res['UCL'].mean()
            print(f"  {ctype}: UCL(평균)={ucl_val:.4f}, CL={res['CL']}, LCL(평균)={res['LCL'].mean():.4f}")

    df_defect = pd.DataFrame({
        'Lot'        : range(1, n_lots + 1),
        'sample_size': np.random.randint(480, 520, n_lots),
        'Defects'    : np.random.poisson(50, n_lots),
    })
    for ctype in ['C', 'U']:
        res = calc_attribute_chart(df_defect, 'Lot', 'sample_size', 'Defects', ctype)
        if res['fixed_limits']:
            print(f"  {ctype}: UCL={res['UCL']}, CL={res['CL']}, LCL={res['LCL']}")
        else:
            print(f"  {ctype}: UCL(평균)={res['UCL'].mean():.4f}, CL={res['CL']}, LCL(평균)={res['LCL'].mean():.4f}")

    # ── 11. Nelson Rule 판정 ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("[11] Nelson Rule 이상 판정")
    # 이상 패턴이 포함된 인위적 데이터 생성
    np.random.seed(1)
    base = np.random.normal(10, 1, 30)
    base[5]  = 14.5   # Rule 1 위반: ±3σ 이탈
    base[15:24] = [10.5] * 9  # Rule 2 위반: 연속 9점 CL 위
    test_series = pd.Series(base, index=range(1, 31))

    UCL_t, CL_t, LCL_t = 13.0, 10.0, 7.0
    nr = apply_nelson_rules(test_series, UCL_t, CL_t, LCL_t)
    print(f"  이상 발생 여부: {nr['any_violation']}")
    for msg in nr['summary']:
        print(msg)

    # ── 12. 공정 자동 해석 ────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[12] 공정능력 자동 해석")
    messages = interpret_capability(cp_res, pp_res)
    for msg in messages:
        print(f"\n{msg}")

    print("\n" + "=" * 65)
    print("  ✅ 모든 함수 테스트 완료")
    print("=" * 65)
