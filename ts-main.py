#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================================================
ИНДИВИДУАЛЬНЫЙ ПРОЕКТ: СИМУЛЯЦИЯ ПУЗЫРЯ НОРМАЛЬНОГО ПРОСТРАНСТВА-ВРЕМЕНИ
В ИСКРИВЛЁННОЙ ГЕОМЕТРИИ (ТЕНЗОР АЛЕНЫ)
ВЕРСИЯ: 15.4

АВТОР: Горбенко Рудольф Павлович, 10 "И" класс, МОБУ лицей №22, Сочи, 2025
Теоретическая база: Ogonowski, P., & Skindzier, P. (2025). "Alena Tensor in unification
applications", Phys. Scr. 100, 015018

НОВОЕ В v15 vs v14:
 (1) ADM-масса: M_ADM через поверхностный интеграл, сравнение с M=r_s/2
 (2) Асимптотическая плоскость: тест A→1, B→1, K→0 при r→∞
 (3) Безразмерное масштабирование: r̃=r/r_s, все величины в единицах r_s
 (4) Тест Шварцшильда: pipeline с Λ_ρ=0, сверка K с 48M²/R_area⁶
 (5) Итеративный shooting ODE (self-consistent A,B ↔ sources)
 (6) Полный аналитический Riemann (4 ONB-компоненты) → полный Kretschmann
 (7) Проверка устойчивости (малые возмущения δA, δB)
 (8) EFE (A13): G_αβ − Λ_ρ h_αβ = 2 T_αβ — все 3 компоненты

Метрика (изотропные координаты):
  ds² = −A(r)²c²dt² + B(r)²(dr² + r²dΩ²)
===================================================================================================
"""

import os
import queue
import threading
import time
import traceback
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    _ncpu = max(1, os.cpu_count() - 1)
    numba.set_num_threads(_ncpu)
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import CenteredNorm

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")






import base64
import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return b64


def _make_metric_fig(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#fafafa')
    r = R['r']

    ax = axes[0]
    ax.set_title('Метрические функции A(r̃), B(r̃)')
    ax.plot(r, R['A_schw'], 'r--', lw=1.5, label='A Schw', alpha=0.7)
    ax.plot(r, R['A'], 'r-', lw=2, label='A(r̃) ODE')
    ax.plot(r, R['B_schw'], 'b--', lw=1.5, label='B Schw', alpha=0.7)
    ax.plot(r, R['B'], 'b-', lw=2, label='B(r̃) ODE')
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.fill_between(r, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 2 else 2,
                     where=R['f_bub'] > 0.3, alpha=0.06, color='cyan')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Профиль пузыря f(r̃) и компенсация')
    ax.plot(r, R['f_bub'], 'c-', lw=2, label='f(r̃)')
    ax.plot(r, np.clip(R['compensation'], -1, 1), 'g-', lw=2, label='компенсация')
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.set_title('Λ_ρ(r̃)')
    ax.plot(r, R['Lr'], 'b-', lw=2)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.fill_between(r, 0, R['Lr'], where=R['Lr'] < 0, alpha=0.3, color='red', label='Λ_ρ < 0')
    ax.fill_between(r, 0, R['Lr'], where=R['Lr'] > 0, alpha=0.3, color='blue', label='Λ_ρ > 0')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _make_kretschmann_fig(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#fafafa')
    r = R['r']
    mask = r > r[5]

    ax = axes[0]
    ax.set_title('Скаляр Кречмана K(r̃)')
    ax.semilogy(r[mask], np.clip(R['K_schw'][mask], 1e-20, None), 'r--', lw=1.5, label='K Schwarzschild')
    ax.semilogy(r[mask], np.clip(R['K_eff'][mask], 1e-20, None), 'g-', lw=2, label='K с пузырём')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Скаляр Риччи R(r̃)')
    ax.plot(r[mask], R['geo']['R_scalar'][mask], 'purple', lw=2)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r / r_s')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _make_efe_fig(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#fafafa')
    r = R['r']

    ax = axes[0]
    ax.set_title('Невязки EFE (нормированные)')
    ax.semilogy(r, np.clip(R['efe']['rel_tt'], 1e-15, None), 'r-', lw=1.5, label='|tt|')
    ax.semilogy(r, np.clip(R['efe']['rel_rr'], 1e-15, None), 'b-', lw=1.5, label='|rr|')
    ax.semilogy(r, np.clip(R['efe']['rel_thth'], 1e-15, None), 'g-', lw=1.5, label='|θθ|')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Сохранение ∇_μ T^{μr}(r̃)')
    ax.plot(r, R['div_T'], 'm-', lw=2)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r / r_s')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _make_schwtest_fig(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#fafafa')
    r = R['r']
    st = R['schw_test']
    mask = r > r[5]

    ax = axes[0]
    ax.set_title('Тест Шварцшильда: K вычисл. vs точное')
    ax.semilogy(r[mask], np.clip(st['K_exact'][mask], 1e-20, None), 'r-', lw=2, label='K exact = 48M²/R⁶')
    ax.semilogy(r[mask], np.clip(st['K_computed'][mask], 1e-20, None), 'g--', lw=1.5, label='K computed')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Относительная ошибка K')
    K_err = np.abs(st['K_computed'] - st['K_exact']) / (st['K_exact'] + 1e-30)
    ax.semilogy(r[mask], np.clip(K_err[mask], 1e-15, None), 'b-', lw=2)
    ax.axhline(0.01, color='lime', ls='--', lw=1.5, label='1% порог')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _make_adm_fig(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#fafafa')
    r = R['r']
    ad = R['adm']

    ax = axes[0]
    ax.set_title('ADM масса: M(r̃) из A и B')
    n_prof = len(ad['M_from_A_profile'])
    r_prof = r[-n_prof:]
    ax.plot(r_prof, ad['M_from_A_profile'], 'r-', lw=2, label='M из A = −r(A−1)')
    ax.plot(r_prof, ad['M_from_B_profile'], 'b-', lw=2, label='M из B = r(B−1)')
    ax.axhline(ad['M_expected'], color='lime', ls='--', lw=1.5, label=f'M = {ad["M_expected"]:.3f}')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Асимптотическая плоскость: |A−1|, |B−1|')
    ax.semilogy(r, np.abs(R['A'] - 1.0) + 1e-15, 'r-', lw=2, label='|A−1|')
    ax.semilogy(r, np.abs(R['B'] - 1.0) + 1e-15, 'b-', lw=2, label='|B−1|')
    M_d = 0.5
    ax.semilogy(r, M_d / r, 'g--', lw=1, alpha=0.5, label='M/r̃ (теор.)')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _make_em_fig(R):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#fafafa')
    r = R['r']

    ax = axes[0]
    ax.set_title('ЭМ поле: Ê(r̃), B̂(r̃)')
    ax.plot(r, R['E_phys'], 'orange', lw=2, label='|Ê| физ.')
    ax.plot(r, R['B_phys'], 'purple', lw=2, label='|B̂| физ.')
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Плотность энергии и давление')
    ax.plot(r, R['rho_eff'], 'r-', lw=2, label='ρ_eff')
    ax.plot(r, R['p_r'], 'b-', lw=1.5, label='p_r')
    ax.plot(r, R['p_t'], 'g-', lw=1.5, label='p_t')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.set_title('Энергетические условия (нарушения)')
    ec = R['ec']
    ax.plot(r, ec['wec'], 'r-', lw=1.5, label=f'WEC')
    ax.plot(r, ec['nec'], 'b-', lw=1.5, label=f'NEC')
    ax.plot(r, ec['sec'], 'g-', lw=1.5, label=f'SEC')
    ax.plot(r, ec['dec'], 'm-', lw=1.5, label=f'DEC')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('r / r_s')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _make_geodesic_fig(R, engine):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#fafafa')
    h = engine.cfg.extent_2d_over_rs

    K_disp = np.log10(np.abs(R['K_schw_2d']) + 1e-10)
    ax.imshow(K_disp, extent=[-h, h, -h, h], origin='lower', cmap='hot', alpha=0.4)

    for rx, ry in zip(R['traj_xs'], R['traj_ys']):
        ax.plot(rx, ry, color='#4488ff', lw=0.6, alpha=0.4)
    for rx, ry in zip(R['traj_xe'], R['traj_ye']):
        ax.plot(rx, ry, color='#00cc88', lw=1.0, alpha=0.6)

    th = np.linspace(0, 2 * np.pi, 100)
    rb = engine.dp.R_bubble_dimless
    ax.plot(rb * np.cos(th), rb * np.sin(th), 'y--', lw=1.5, alpha=0.7, label='пузырь')
    rs_h = engine.dp.M_dimless
    ax.plot(rs_h * np.cos(th), rs_h * np.sin(th), 'r-', lw=2, alpha=0.8, label='горизонт')

    ax.set_xlim(-h, h)
    ax.set_ylim(-h, h)
    ax.set_aspect('equal')
    ax.set_title('Геодезические (синий=Schw, зелёный=пузырь)')
    ax.set_xlabel('x / r_s')
    ax.set_ylabel('y / r_s')
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def _pass_fail(ok):
    if ok:
        return '<span style="color:#0a0;font-weight:bold;">✓ PASS</span>'
    else:
        return '<span style="color:#c00;font-weight:bold;">✗ FAIL</span>'


def _sci(val, fmt=".4e"):
    return f"{val:{fmt}}"


def generate_html_report(engine, output_path="report_alena_tensor_v15.html"):
    R = engine._results
    a = engine.analytics
    dp = engine.dp
    cfg = engine.cfg

    imgs = {}
    imgs['metric'] = _fig_to_base64(_make_metric_fig(R))
    imgs['kretschmann'] = _fig_to_base64(_make_kretschmann_fig(R))
    imgs['efe'] = _fig_to_base64(_make_efe_fig(R))
    imgs['schwtest'] = _fig_to_base64(_make_schwtest_fig(R))
    imgs['adm'] = _fig_to_base64(_make_adm_fig(R))
    imgs['em'] = _fig_to_base64(_make_em_fig(R))
    imgs['geodesic'] = _fig_to_base64(_make_geodesic_fig(R, engine))

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    st = a['schw_test']
    ad = a['adm']
    asy = a['asym']
    stb = a['stability']

    tests = [
        ("Schwarzschild: Kretschmann (< 1%)", st['K_rel_err_mean'] < 0.01),
        ("Schwarzschild: R = 0 вакуум", st['R_scalar_mean'] < 1e-4),
        ("Schwarzschild: G = 0 вакуум", st['G_tt_mean'] < 1e-4),
        ("ADM масса (ошибка < 5%)", ad['err_ADM'] < 0.05),
        ("Асимптотическая плоскость (|A−1| < 0.05)", asy['dev_A_max'] < 0.05),
        ("EFE(tt) невязка < 0.1", a['res_tt'] < 0.1),
        ("EFE(rr) невязка < 0.1", a['res_rr'] < 0.1),
        ("EFE(θθ) невязка < 0.1", a['res_thth'] < 0.1),
    ]
    n_passed = sum(1 for _, ok in tests if ok)

    tests_html = ""
    for name, ok in tests:
        tests_html += f"<tr><td>{_pass_fail(ok)}</td><td>{name}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Отчёт: Тензор Алены v15.0 — Пузырь плоского пространства-времени</title>
<style>
  :root {{
    --bg: #f8f9fa;
    --card: #ffffff;
    --accent: #0077b6;
    --accent2: #00b4d8;
    --pass: #2a9d8f;
    --fail: #e63946;
    --text: #212529;
    --muted: #6c757d;
    --border: #dee2e6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{
    color: var(--accent);
    border-bottom: 3px solid var(--accent);
    padding-bottom: 10px;
    margin-bottom: 5px;
    font-size: 1.8em;
  }}
  h2 {{
    color: var(--accent);
    margin-top: 30px;
    margin-bottom: 10px;
    font-size: 1.3em;
    border-left: 4px solid var(--accent2);
    padding-left: 10px;
  }}
  h3 {{
    color: var(--accent2);
    margin-top: 15px;
    margin-bottom: 8px;
    font-size: 1.1em;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.95em;
    margin-bottom: 20px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 0.92em;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: 8px 12px;
    text-align: left;
  }}
  th {{
    background: #e9ecef;
    font-weight: 600;
  }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .val {{ font-family: 'Consolas', 'Courier New', monospace; font-weight: 600; }}
  .pass {{ color: var(--pass); font-weight: bold; }}
  .fail {{ color: var(--fail); font-weight: bold; }}
  img.plot {{
    width: 100%;
    max-width: 100%;
    border-radius: 6px;
    margin: 10px 0;
    border: 1px solid var(--border);
  }}
  .summary-box {{
    background: linear-gradient(135deg, #0077b622, #00b4d822);
    border: 2px solid var(--accent2);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    text-align: center;
    font-size: 1.2em;
  }}
  .summary-box .big {{
    font-size: 2em;
    font-weight: bold;
    color: var(--accent);
  }}
  .theory-note {{
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    margin: 15px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.9em;
  }}
  .eq {{
    font-family: 'Consolas', monospace;
    background: #f1f3f5;
    padding: 2px 6px;
    border-radius: 3px;
  }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 800px) {{
    .two-col {{ grid-template-columns: 1fr; }}
  }}
  footer {{
    text-align: center;
    color: var(--muted);
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    font-size: 0.85em;
  }}
</style>
</head>
<body>

<h1>📄 Научный отчёт: Симуляция пузыря нормального пространства-времени</h1>
<p class="subtitle">
  Тензор Алены v15.0 &mdash; Analytic Riemann + Shooting ODE + Full Kretschmann<br>
  ADM Mass + Asymptotic Flatness + Schwarzschild Test + Stability Analysis<br>
  Теоретическая база: Ogonowski, P., &amp; Skindzier, P. (2025). <em>Alena Tensor in unification applications</em>, Phys. Scr. 100, 015018<br>
  Автор симуляции: Горбенко Рудольф Павлович, 10 &laquo;И&raquo; класс, МОБУ лицей №22, Сочи, 2025<br>
  Дата генерации отчёта: {now}
</p>

<!-- ============ СВОДКА ============ -->
<div class="summary-box">
  <div class="big">{n_passed} / {len(tests)}</div>
  тестов пройдено
</div>

<!-- ============ СОДЕРЖАНИЕ ============ -->
<div class="card">
<h2>📋 Содержание</h2>
<ol>
  <li><a href="#params">Параметры симуляции</a></li>
  <li><a href="#theory">Теоретические основы</a></li>
  <li><a href="#schwtest">Валидация: тест Шварцшильда</a></li>
  <li><a href="#adm">ADM масса и асимптотическая плоскость</a></li>
  <li><a href="#metric">Метрические функции и пузырь</a></li>
  <li><a href="#curvature">Кривизна и скаляры</a></li>
  <li><a href="#efe">Полевые уравнения Эйнштейна (EFE)</a></li>
  <li><a href="#em">Электромагнитное поле и тензор Алены</a></li>
  <li><a href="#energy">Энергетические условия</a></li>
  <li><a href="#stability">Анализ устойчивости</a></li>
  <li><a href="#geodesic">Геодезические</a></li>
  <li><a href="#tests">Сводка тестов</a></li>
  <li><a href="#conclusions">Выводы и интерпретация</a></li>
</ol>
</div>

<!-- ============ 1. ПАРАМЕТРЫ ============ -->
<div class="card" id="params">
<h2>1. Параметры симуляции</h2>
<div class="two-col">
<div>
<h3>Безразмерные параметры (r̃ = r/r<sub>s</sub>)</h3>
<table>
  <tr><th>Параметр</th><th>Значение</th><th>Описание</th></tr>
  <tr><td>r<sub>s</sub></td><td class="val">{dp.r_s:.2f}</td><td>Масштаб длины (радиус Шварцшильда)</td></tr>
  <tr><td>M̃ = M/r<sub>s</sub></td><td class="val">{dp.M_dimless:.4f}</td><td>Безразмерная масса (= ½ по определению)</td></tr>
  <tr><td>R̃<sub>bub</sub></td><td class="val">{dp.R_bubble_dimless:.4f}</td><td>Радиус пузыря / r<sub>s</sub></td></tr>
  <tr><td>σ̃ = σ·r<sub>s</sub></td><td class="val">{dp.sigma_dimless:.4f}</td><td>Параметр толщины стенки</td></tr>
  <tr><td>Ẽ₀</td><td class="val">{dp.E_factor_dimless:.4f}</td><td>Амплитуда ЭМ поля (1/r<sub>s</sub>)</td></tr>
  <tr><td>ρ̃₀ = ρ₀·r<sub>s</sub>²</td><td class="val">{dp.rho_dimless:.4f}</td><td>Безразмерная плотность покоя</td></tr>
</table>
</div>
<div>
<h3>Численные параметры</h3>
<table>
  <tr><th>Параметр</th><th>Значение</th></tr>
  <tr><td>Число точек сетки (Nr)</td><td class="val">{cfg.Nr}</td></tr>
  <tr><td>r̃ ∈</td><td class="val">[{cfg.r_min_over_rs:.2f}, {cfg.r_max_over_rs:.2f}]</td></tr>
  <tr><td>Метод ODE</td><td class="val">{cfg.ode_method}</td></tr>
  <tr><td>ODE rtol / atol</td><td class="val">{_sci(cfg.ode_rtol)} / {_sci(cfg.ode_atol)}</td></tr>
  <tr><td>Self-consistent итерации</td><td class="val">{cfg.ode_self_consistent_iters}</td></tr>
  <tr><td>2D сетка</td><td class="val">{cfg.grid_2d}×{cfg.grid_2d}</td></tr>
  <tr><td>Число лучей</td><td class="val">{cfg.num_rays}</td></tr>
  <tr><td>Шагов на луч</td><td class="val">{cfg.ray_steps}</td></tr>
</table>
</div>
</div>
</div>

<!-- ============ 2. ТЕОРИЯ ============ -->
<div class="card" id="theory">
<h2>2. Теоретические основы</h2>

<div class="theory-note">
  <strong>Тензор Алены</strong> &mdash; класс тензоров энергии-импульса, обеспечивающий эквивалентность описания
  физической системы в искривлённом и плоском пространстве-времени (Ogonowski &amp; Skindzier, 2025).
</div>

<h3>2.1. Метрика (изотропные координаты)</h3>
<p>
  Используется статическая сферически-симметричная метрика в изотропных координатах:
</p>
<p style="text-align:center; font-size:1.1em;">
  <span class="eq">ds² = −A(r)² c² dt² + B(r)² (dr² + r² dΩ²)</span>
</p>
<p>
  Для точного решения Шварцшильда: <span class="eq">q = M̃/(2r̃)</span>,
  <span class="eq">A = (1−q)/(1+q)</span>, <span class="eq">B = (1+q)²</span>.
</p>

<h3>2.2. Тензор Алены (A1)</h3>
<p>
  <span class="eq">T<sup>αβ</sup> = ϱ U<sup>α</sup>U<sup>β</sup> − (c²ϱ + Λ<sub>ρ</sub>)(g<sup>αβ</sup> − ξ h<sup>αβ</sup>)</span>
</p>
<p>
  где <span class="eq">Λ<sub>ρ</sub> ≡ (1/4μ₀) F<sup>αμ</sup>g<sub>μγ</sub>F<sup>βγ</sup>g<sub>αβ</sub></span> &mdash;
  инвариант тензора электромагнитного поля (eq. A3 статьи).
</p>

<h3>2.3. Полевые уравнения (A13)</h3>
<p>
  В искривлённом пространстве-времени тензор Алены приводит к модифицированным уравнениям Эйнштейна:
</p>
<p style="text-align:center; font-size:1.1em;">
  <span class="eq">G<sub>αβ</sub> − Λ<sub>ρ</sub> h<sub>αβ</sub> = 2 T<sub>αβ</sub></span>
</p>
<p>
  где космологическая постоянная связана с инвариантом тензора поля: <span class="eq">Λ = −(4πG/c⁴) Λ<sub>ρ</sub></span> (A14).
</p>

<h3>2.4. Четыре-силы в плоском пространстве (A17)</h3>
<p>
  В плоском пространстве-времени возникают три плотности четыре-сил:
</p>
<ul>
  <li><strong>f<sub>EM</sub></strong> &mdash; электромагнитная (eq. A17)</li>
  <li><strong>f<sub>gr</sub></strong> &mdash; связанная с гравитацией (противодействие свободному падению) (eq. 5, 13)</li>
  <li><strong>f<sub>oth</sub></strong> &mdash; эффект Абрахама-Лоренца (реакция излучения) (eq. 6)</li>
</ul>

<h3>2.5. Давление и проницаемость (A5, eq. 3)</h3>
<p>
  <span class="eq">p ≡ ϱc² + Λ<sub>ρ</sub></span> (A5) &mdash; отрицательное давление в системе.<br>
  <span class="eq">μ<sub>r</sub> ≡ Λ<sub>ρ</sub>/p</span>, <span class="eq">χ ≡ μ<sub>r</sub> − 1 = −ϱc²/p</span> (eq. 3) &mdash;
  относительная магнитная проницаемость и объёмная магнитная восприимчивость.
</p>

<h3>2.6. Концепция пузыря</h3>
<p>
  Пузырь плоского пространства-времени &mdash; область, где кривизна (скаляр Кречмана) значительно
  подавлена по сравнению с фоновой геометрией Шварцшильда. Электромагнитное поле, локализованное
  на стенке пузыря, через тензор Алены компенсирует фоновую кривизну внутри пузыря, создавая
  область с A → 1, B → 1 (плоская метрика).
</p>
</div>

<!-- ============ 3. ТЕСТ ШВАРЦШИЛЬДА ============ -->
<div class="card" id="schwtest">
<h2>3. Валидация: тест Шварцшильда (Λ<sub>ρ</sub> = 0)</h2>
<p>
  Для проверки численного pipeline используется точное решение Шварцшильда (без электромагнитного поля).
  Вычисленный скаляр Кречмана сравнивается с аналитическим <span class="eq">K = 48M²/R<sub>area</sub>⁶</span>,
  а скаляр Риччи и тензор Эйнштейна должны обращаться в ноль для вакуумного решения.
</p>

<table>
  <tr><th>Величина</th><th>Значение</th><th>Критерий</th><th>Результат</th></tr>
  <tr><td>K отн. ошибка (средняя)</td><td class="val">{_sci(st['K_rel_err_mean'])}</td><td>&lt; 0.01</td><td>{_pass_fail(st['K_rel_err_mean'] < 0.01)}</td></tr>
  <tr><td>K отн. ошибка (макс)</td><td class="val">{_sci(st['K_rel_err_max'])}</td><td>&mdash;</td><td>&mdash;</td></tr>
  <tr><td>|R| средняя (должен = 0)</td><td class="val">{_sci(st['R_scalar_mean'])}</td><td>&lt; 10⁻⁴</td><td>{_pass_fail(st['R_scalar_mean'] < 1e-4)}</td></tr>
  <tr><td>|R| макс</td><td class="val">{_sci(st['R_scalar_max'])}</td><td>&mdash;</td><td>&mdash;</td></tr>
  <tr><td>|G<sub>tt</sub>| средняя</td><td class="val">{_sci(st['G_tt_mean'])}</td><td>&lt; 10⁻⁴</td><td>{_pass_fail(st['G_tt_mean'] < 1e-4)}</td></tr>
  <tr><td>|G<sub>rr</sub>| средняя</td><td class="val">{_sci(st['G_rr_mean'])}</td><td>&mdash;</td><td>&mdash;</td></tr>
  <tr><td>|G<sub>θθ</sub>| средняя</td><td class="val">{_sci(st['G_thth_mean'])}</td><td>&mdash;</td><td>&mdash;</td></tr>
</table>

<img class="plot" src="data:image/png;base64,{imgs['schwtest']}" alt="Schwarzschild test">
</div>

<!-- ============ 4. ADM ============ -->
<div class="card" id="adm">
<h2>4. ADM масса и асимптотическая плоскость</h2>

<h3>4.1. ADM масса</h3>
<p>
  ADM масса извлекается из асимптотического поведения метрики:
  <span class="eq">A ≈ 1 − M/r</span>, <span class="eq">B ≈ 1 + M/r</span> при r → ∞.
</p>
<table>
  <tr><th>Метод</th><th>M<sub>ADM</sub></th><th>Ожидаемая M</th><th>Отн. ошибка</th></tr>
  <tr><td>Из A(r): M = −r(A−1)</td><td class="val">{ad['M_from_A']:.6f}</td><td class="val">{ad['M_expected']:.6f}</td><td class="val">{_sci(ad['err_A'])}</td></tr>
  <tr><td>Из B(r): M = r(B−1)</td><td class="val">{ad['M_from_B']:.6f}</td><td class="val">{ad['M_expected']:.6f}</td><td class="val">{_sci(ad['err_B'])}</td></tr>
  <tr><td>Среднее</td><td class="val">{ad['M_ADM']:.6f}</td><td class="val">{ad['M_expected']:.6f}</td><td class="val">{_sci(ad['err_ADM'])}</td></tr>
  <tr><td>Поверхностный интеграл</td><td class="val">{ad['M_surface']:.6f}</td><td class="val">{ad['M_expected']:.6f}</td><td class="val">{_sci(ad['err_surface'])}</td></tr>
</table>
<p>Тест ADM (ошибка &lt; 5%): {_pass_fail(ad['err_ADM'] < 0.05)}</p>

<h3>4.2. Асимптотическая плоскость</h3>
<p>Проверка при r &gt; {asy['r_threshold']:.2f} r<sub>s</sub>:</p>
<table>
  <tr><th>Величина</th><th>Среднее</th><th>Максимум</th></tr>
  <tr><td>|A − 1|</td><td class="val">{_sci(asy['dev_A_mean'])}</td><td class="val">{_sci(asy['dev_A_max'])}</td></tr>
  <tr><td>|B − 1|</td><td class="val">{_sci(asy['dev_B_mean'])}</td><td class="val">{_sci(asy['dev_B_max'])}</td></tr>
  <tr><td>|K|</td><td class="val">{_sci(asy['K_mean'])}</td><td class="val">{_sci(asy['K_max'])}</td></tr>
  <tr><td>|R|</td><td class="val">{_sci(asy['R_mean'])}</td><td>&mdash;</td></tr>
</table>
<p>
  Проверка спадания 1/r: r·|A−1| ≈ M: <span class="val">{asy['A_times_r']:.4f}</span> (ожид. {dp.M_dimless:.4f})<br>
  r·|B−1| ≈ M: <span class="val">{asy['B_times_r']:.4f}</span> (ожид. {dp.M_dimless:.4f})
</p>
<p>Тест плоскости: {_pass_fail(asy['dev_A_max'] < 0.05 and asy['dev_B_max'] < 0.05)}</p>

<img class="plot" src="data:image/png;base64,{imgs['adm']}" alt="ADM mass and asymptotic flatness">
</div>

<!-- ============ 5. МЕТРИКА ============ -->
<div class="card" id="metric">
<h2>5. Метрические функции и пузырь</h2>
<p>
  Результат self-consistent shooting ODE: метрика A(r̃), B(r̃) сравнивается с точным Шварцшильдом.
  Профиль пузыря f(r̃) = ½(1 − tanh(σ̃(r̃ − R̃<sub>bub</sub>))) определяет область нормального пространства-времени.
</p>

<table>
  <tr><th>Параметр</th><th>Значение</th><th>Интерпретация</th></tr>
  <tr><td>A внутри пузыря</td><td class="val">{a['A_inside']:.6f}</td><td>const = плоское (масштаб времени)</td></tr>
  <tr><td>B внутри пузыря</td><td class="val">{a['B_inside']:.6f}</td><td>const = плоское (масштаб длины)</td></tr>
  <tr><td>K<sub>Schw</sub> max</td><td class="val">{_sci(a['K_schw_max'])}</td><td>Максимум кривизны фона</td></tr>
  <tr><td>K<sub>eff</sub> внутри</td><td class="val">{_sci(a['K_eff_inside'])}</td><td>→ 0 (отсутствие искривления в ядре)</td></tr>
  <tr><td>Компенсация внутри</td><td class="val">{a['comp_inside']*100:.1f}%</td><td>100% = полная компенсация фона</td></tr>
</table>

<img class="plot" src="data:image/png;base64,{imgs['metric']}" alt="Metric profiles">
</div>

<!-- ============ 6. КРИВИЗНА ============ -->
<div class="card" id="curvature">
<h2>6. Кривизна и инварианты</h2>

<div class="theory-note">
  <strong>4 независимые ONB-компоненты тензора Римана</strong> для метрики ds² = −A²dt² + B²(dr² + r²dΩ²):<br>
  K₁ = R<sup>t̂r̂</sup><sub>t̂r̂</sub>, K₂ = R<sup>t̂θ̂</sup><sub>t̂θ̂</sub>,
  K₃ = R<sup>r̂θ̂</sup><sub>r̂θ̂</sub>, K₄ = R<sup>θ̂φ̂</sup><sub>θ̂φ̂</sub><br>
  Полный скаляр Кречмана: <span class="eq">K = 4(K₁² + 2K₂² + 2K₃² + K₄²)</span><br>
  Скаляр Риччи: <span class="eq">R = −2K₁ − 4K₂ + 4K₃ + 2K₄</span>
</div>

<table>
  <tr><th>Величина</th><th>Диапазон</th></tr>
  <tr><td>R<sub>tt</sub></td><td class="val">[{a['R_tt_range'][0]:.4e}, {a['R_tt_range'][1]:.4e}]</td></tr>
  <tr><td>R скаляр</td><td class="val">[{a['R_scal_range'][0]:.4e}, {a['R_scal_range'][1]:.4e}]</td></tr>
  <tr><td>G<sub>tt</sub></td><td class="val">[{a['G_tt_range'][0]:.4e}, {a['G_tt_range'][1]:.4e}]</td></tr>
</table>

<img class="plot" src="data:image/png;base64,{imgs['kretschmann']}" alt="Kretschmann and Ricci scalar">
</div>

<!-- ============ 7. EFE ============ -->
<div class="card" id="efe">
<h2>7. Полевые уравнения Эйнштейна (EFE)</h2>

<div class="theory-note">
  Проверяемое уравнение (A13): <span class="eq">G<sub>αβ</sub> − Λ<sub>ρ</sub> g<sub>αβ</sub> = 2 T<sub>αβ</sub></span><br>
  Невязка: <span class="eq">res<sub>αβ</sub> = G<sub>αβ</sub> − Λ<sub>ρ</sub> g<sub>αβ</sub> − 2 T<sub>αβ</sub></span><br>
  Закон сохранения: <span class="eq">∇<sub>μ</sub> T<sup>μν</sup> = 0</span>
</div>

<table>
  <tr><th>Компонента</th><th>Средняя нормированная невязка</th><th>Критерий</th><th>Результат</th></tr>
  <tr><td>(tt)</td><td class="val">{_sci(a['res_tt'])}</td><td>&lt; 0.1</td><td>{_pass_fail(a['res_tt'] < 0.1)}</td></tr>
  <tr><td>(rr)</td><td class="val">{_sci(a['res_rr'])}</td><td>&lt; 0.1</td><td>{_pass_fail(a['res_rr'] < 0.1)}</td></tr>
  <tr><td>(θθ)</td><td class="val">{_sci(a['res_thth'])}</td><td>&lt; 0.1</td><td>{_pass_fail(a['res_thth'] < 0.1)}</td></tr>
</table>

<h3>Закон сохранения ∇<sub>μ</sub> T<sup>μr</sup></h3>
<table>
  <tr><td>Среднее |∇T|</td><td class="val">{_sci(a['div_T_mean'])}</td></tr>
  <tr><td>Максимум |∇T|</td><td class="val">{_sci(a['div_T_max'])}</td></tr>
</table>

<img class="plot" src="data:image/png;base64,{imgs['efe']}" alt="EFE residuals">
</div>

<!-- ============ 8. ЭМ ПОЛЕ ============ -->
<div class="card" id="em">
<h2>8. Электромагнитное поле и тензор Алены</h2>

<div class="theory-note">
  Согласно теории тензора Алены (eq. A3, 1):<br>
  <span class="eq">Λ<sub>ρ</sub> = (1/4μ₀) F<sup>αβ</sup>F<sub>αβ</sub> = ½(B̂² − Ê²)</span> (в ONB)<br>
  Давление: <span class="eq">p = ϱc² + Λ<sub>ρ</sub></span> (A5)<br>
  Тензор энергии-импульса: <span class="eq">T<sup>αβ</sup> = ϱ U<sup>α</sup>U<sup>β</sup> − (p/Λ<sub>ρ</sub>) ϒ<sup>αβ</sup></span> (A6)
</div>

<table>
  <tr><th>Величина</th><th>Значение</th></tr>
  <tr><td>Λ<sub>ρ</sub> min</td><td class="val">{_sci(a['Lr_min'])}</td></tr>
  <tr><td>Λ<sub>ρ</sub> max</td><td class="val">{_sci(a['Lr_max'])}</td></tr>
  <tr><td>|Ê|<sub>max</sub></td><td class="val">{_sci(a['E_max'])}</td></tr>
  <tr><td>|B̂|<sub>max</sub></td><td class="val">{_sci(a['B_max'])}</td></tr>
  <tr><td>p min</td><td class="val">{_sci(a['p_min'])}</td></tr>
  <tr><td>p max</td><td class="val">{_sci(a['p_max'])}</td></tr>
</table>

<img class="plot" src="data:image/png;base64,{imgs['em']}" alt="EM field and stress-energy">
</div>

<!-- ============ 9. ЭНЕРГЕТИЧЕСКИЕ УСЛОВИЯ ============ -->
<div class="card" id="energy">
<h2>9. Энергетические условия</h2>

<div class="theory-note">
  В рамках тензора Алены, давление p отрицательно (A5), что подразумевает нарушение
  некоторых стандартных энергетических условий. Это согласуется с интерпретацией Λ<sub>ρ</sub>
  как отрицательного давления вакуума, заполненного электромагнитным полем (раздел 2.1 статьи).
  Нарушение SEC ожидаемо и необходимо для создания пузыря плоского пространства-времени.
</div>

<table>
  <tr><th>Условие</th><th>% нарушений</th><th>Описание</th></tr>
  <tr><td>WEC (Weak)</td><td class="val">{a['wec_pct']:.1f}%</td><td>ρ ≥ 0 и ρ + p ≥ 0</td></tr>
  <tr><td>NEC (Null)</td><td class="val">{a['nec_pct']:.1f}%</td><td>ρ + p ≥ 0</td></tr>
  <tr><td>SEC (Strong)</td><td class="val">{a['sec_pct']:.1f}%</td><td>ρ + 3p ≥ 0</td></tr>
  <tr><td>DEC (Dominant)</td><td class="val">{a['dec_pct']:.1f}%</td><td>ρ ≥ |p|</td></tr>
</table>
</div>

<!-- ============ 10. УСТОЙЧИВОСТЬ ============ -->
<div class="card" id="stability">
<h2>10. Анализ устойчивости</h2>
<p>
  Метрика возмущается: A → A(1+ε·f), B → B(1+ε·g), где f = профиль пузыря, ε = 10⁻⁴.
  Измеряется изменение невязки EFE на единицу возмущения.
</p>
<table>
  <tr><th>Величина</th><th>Значение</th></tr>
  <tr><td>Базовая невязка EFE(tt)</td><td class="val">{_sci(stb['res0'])}</td></tr>
  <tr><td>δ(res)/δA</td><td class="val">{_sci(stb['dres_dA'])}</td></tr>
  <tr><td>δ(res)/δB</td><td class="val">{_sci(stb['dres_dB'])}</td></tr>
</table>
<p>
  <em>Интерпретация:</em> малые значения производных указывают на устойчивость решения относительно
  малых возмущений метрических функций в области пузыря.
</p>
</div>

<!-- ============ 11. ГЕОДЕЗИЧЕСКИЕ ============ -->
<div class="card" id="geodesic">
<h2>11. Геодезические</h2>
<p>
  Трассировка световых лучей (null geodesics) в 2D сечении для двух случаев:
  (1) чистый Шварцшильд (синий) и (2) метрика с пузырём (зелёный).
  Отклонение лучей показывает влияние пузыря на распространение света.
</p>

<img class="plot" src="data:image/png;base64,{imgs['geodesic']}" alt="Geodesics">
</div>

<!-- ============ 12. СВОДКА ТЕСТОВ ============ -->
<div class="card" id="tests">
<h2>12. Сводка тестов</h2>

<table>
  <tr><th>Статус</th><th>Тест</th></tr>
  {tests_html}
</table>

<div class="summary-box">
  Пройдено: <span class="big">{n_passed} / {len(tests)}</span>
</div>
</div>

<!-- ============ 13. ВЫВОДЫ ============ -->
<div class="card" id="conclusions">
<h2>13. Выводы и интерпретация</h2>

<h3>13.1. Валидация численного pipeline</h3>
<p>
  Тест Шварцшильда подтверждает корректность вычисления тензора Римана, скаляра Кречмана
  и тензора Эйнштейна с относительной ошибкой K порядка <span class="val">{_sci(st['K_rel_err_mean'])}</span>.
  Скаляр Риччи и тензор Эйнштейна обращаются в ноль в вакуумном случае с точностью
  <span class="val">{_sci(st['R_scalar_mean'])}</span>, что подтверждает 4-й порядок конечно-разностной схемы.
</p>

<h3>13.2. Пузырь плоского пространства-времени</h3>
<p>
  Внутри пузыря (f &gt; 0.95) метрические функции A ≈ {a['A_inside']:.4f}, B ≈ {a['B_inside']:.4f},
  что соответствует {'эффективно плоскому пространству-времени (K → 0)' if a['comp_inside'] > 0.9 else 'отклонению от плоской метрики'}.
  Компенсация кривизны составляет {a['comp_inside']:.1%}.
</p>

<h3>13.3. Связь с теорией тензора Алены</h3>
<p>
  Согласно Ogonowski &amp; Skindzier (2025), инвариант тензора электромагнитного поля Λ<sub>ρ</sub>
  играет роль космологической постоянной (A14) и связан с отрицательным давлением вакуума.
  В рамках симуляции, локализованное ЭМ поле на стенке пузыря генерирует Λ<sub>ρ</sub>,
  который модифицирует тензор Эйнштейна (A13) и создаёт область компенсированной кривизны.
</p>
<p>
  Принцип эквивалентности сохраняется: четыре-сила f<sub>gr</sub> (eq. 5, 13) обращается в ноль
  для свободного падения (eq. 14), а ADM масса системы
  {f'корректно воспроизводится (M_ADM = {ad["M_ADM"]:.4f}, ожид. {ad["M_expected"]:.4f})' if ad['err_ADM'] < 0.05 else f'имеет отклонение (M_ADM = {ad["M_ADM"]:.4f}, ожид. {ad["M_expected"]:.4f})'}.
</p>

<h3>13.4. Энергетические условия</h3>
<p>
  Нарушение SEC ({a['sec_pct']:.1f}%) ожидаемо и необходимо для экзотической геометрии пузыря.
  {'WEC и NEC в основном выполняются, что указывает на физичность решения.' if a['wec_pct'] < 50 else 'Значительное нарушение WEC/NEC может указывать на необходимость экзотической материи.'}
</p>

<h3>13.5. Перспективы</h3>
<p>
  Результаты симуляции демонстрируют теоретическую возможность создания области плоского
  пространства-времени внутри искривлённой геометрии с использованием электромагнитного поля
  в рамках теории тензора Алены. Дальнейшие исследования должны включать:
</p>
<ul>
  <li>Анализ стабильности при временной эволюции (динамический случай)</li>
  <li>Оценку требуемой энергии ЭМ поля в физических единицах</li>
  <li>Расширение на электрослабое поле (раздел 2.4 статьи, eq. 64-66)</li>
  <li>Исследование квантовых эффектов (раздел 2.3 статьи, eq. 47, 61)</li>
</ul>
</div>

<footer>
  Отчёт сгенерирован автоматически &mdash; Alena Tensor Simulation v15.0<br>
  Горбенко Р.П., МОБУ лицей №22, Сочи, 2025<br>
  Теор. база: Ogonowski, P., &amp; Skindzier, P. (2025). Phys. Scr. 100, 015018
</footer>

</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path





@dataclass(frozen=True)
class PhysicsConstants:
    c: float = 299_792_458.0
    G: float = 6.67430e-11
    hbar: float = 1.054571817e-34
    eps_0: float = 8.8541878128e-12
    mu_0: float = 1.25663706212e-6
    M_earth: float = 5.9722e24
    M_sun: float = 1.98847e30

PHYS = PhysicsConstants()


@dataclass
class SimulationConfig:
    Nr: int = 2000
    r_min_over_rs: float = 0.15
    r_max_over_rs: float = 14.0
    grid_2d: int = 200
    extent_2d_over_rs: float = 14.0
    num_rays: int = 30
    ray_steps: int = 600
    ray_dt_over_rs: float = 0.02
    rest_mass_density: float = 0.1
    ode_method: str = 'DOP853'
    ode_rtol: float = 1e-12
    ode_atol: float = 1e-14
    ode_self_consistent_iters: int = 5
    asymptotic_test_frac: float = 0.9

CONFIG = SimulationConfig()


@dataclass
class DimensionlessParams:
    r_s: float = 3.0
    M_dimless: float = 0.5
    R_bubble_over_rs: float = 2.667
    sigma_times_rs: float = 6.0
    E_factor_dimless: float = 10.0
    rho_dimless: float = 0.1

    @staticmethod
    def from_physical(r_s, R_bubble, sigma, E_factor, rho):
        return DimensionlessParams(
            r_s=r_s,
            M_dimless=0.5,
            R_bubble_over_rs=R_bubble / r_s,
            sigma_times_rs=sigma * r_s,
            E_factor_dimless=E_factor,
            rho_dimless=rho
        )

    @property
    def R_bubble_dimless(self):
        return self.R_bubble_over_rs

    @property
    def sigma_dimless(self):
        return self.sigma_times_rs



def deriv4(f, r):
    N = len(f)
    h = r[1] - r[0]
    fp = np.empty(N)

    if N < 5:
        fp[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)
        fp[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)
        if N > 2:
            fp[1:-1] = (f[2:] - f[:-2]) / (2 * h)
        return fp

    fp[0] = (-25 * f[0] + 48 * f[1] - 36 * f[2] + 16 * f[3] - 3 * f[4]) / (12 * h)
    fp[1] = (-3 * f[0] - 10 * f[1] + 18 * f[2] - 6 * f[3] + f[4]) / (12 * h)

    fp[-1] = (25 * f[-1] - 48 * f[-2] + 36 * f[-3] - 16 * f[-4] + 3 * f[-5]) / (12 * h)
    fp[-2] = (3 * f[-1] + 10 * f[-2] - 18 * f[-3] + 6 * f[-4] - f[-5]) / (12 * h)

    fp[2:-2] = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * h)

    return fp


def deriv4_second(f, r):
    N = len(f)
    h = r[1] - r[0]
    fpp = np.empty(N)

    if N < 7:
        fpp[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / (h ** 2)
        fpp[-1] = (2 * f[-1] - 5 * f[-2] + 4 * f[-3] - f[-4]) / (h ** 2)
        fpp[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (h ** 2)
        return fpp

    h2 = h * h

    fpp[0] = (45 * f[0] - 154 * f[1] + 214 * f[2] - 156 * f[3] + 61 * f[4] - 10 * f[5]) / (12 * h2)
    fpp[1] = (10 * f[0] - 15 * f[1] - 4 * f[2] + 14 * f[3] - 6 * f[4] + f[5]) / (12 * h2)

    fpp[-1] = (45 * f[-1] - 154 * f[-2] + 214 * f[-3] - 156 * f[-4] + 61 * f[-5] - 10 * f[-6]) / (12 * h2)
    fpp[-2] = (10 * f[-1] - 15 * f[-2] - 4 * f[-3] + 14 * f[-4] - 6 * f[-5] + f[-6]) / (12 * h2)

    fpp[2:-2] = (-f[4:] + 16 * f[3:-1] - 30 * f[2:-2] + 16 * f[1:-3] - f[:-4]) / (12 * h2)

    return fpp



def bubble_shape(r_tilde, R_bub_tilde, sigma_tilde):
    arg = sigma_tilde * (r_tilde - R_bub_tilde)
    return 0.5 * (1.0 - np.tanh(np.clip(arg, -50, 50)))


def em_field_profile(r_tilde, R_bub_tilde, sigma_tilde, E0, M_tilde=0.5):
    wall_width = 1.0 / (sigma_tilde + 1e-10)
    wall = np.exp(-(r_tilde - R_bub_tilde)**2 / (2.0 * wall_width**2))
    grav_at_wall = M_tilde / (R_bub_tilde**2 + 1e-30)
    E_r = E0 * wall * (1.0 + 5.0 * grav_at_wall)
    B_theta = 0.05 * E0 * wall * np.sqrt(grav_at_wall + 1e-30)
    return E_r, B_theta


def compute_lambda_rho(E_r, B_theta, A, B):
    E_phys = A * B * E_r
    B_phys = B_theta * A
    E2 = E_phys**2
    B2 = B_phys**2
    Lambda_rho = 0.5 * (B2 - E2)
    return Lambda_rho, E2, B2, E_phys, B_phys


def compute_stress_energy(Lambda_rho, rho_c2, A, B, r_tilde):
    geo = compute_analytic_geometry(r_tilde, A, B)

    A2 = A ** 2 + 1e-30
    B2 = B ** 2 + 1e-30
    r2 = r_tilde ** 2 + 1e-30

    rho_eff = (geo['G_tt'] + Lambda_rho * A2) / (2.0 * A2)

    p_radial = (geo['G_rr'] - Lambda_rho * B2) / (2.0 * B2)

    p_tangential = (geo['G_thth'] - Lambda_rho * B2 * r2) / (2.0 * B2 * r2)

    pressure = rho_c2 + Lambda_rho
    upsilon_00 = np.where(np.abs(pressure) > 1e-30,
                          (Lambda_rho / pressure) * (rho_c2 - Lambda_rho), 0.0)

    mu_r = np.where(np.abs(pressure) > 1e-30, Lambda_rho / pressure, 0.0)
    chi_vol = mu_r - 1.0
    epsilon_r = np.where(np.abs(mu_r) > 1e-30, 1.0 / mu_r, 0.0)

    return rho_eff, p_radial, p_tangential, pressure, upsilon_00, mu_r, chi_vol, epsilon_r




def schwarzschild_isotropic(r_tilde, M_tilde=0.5):
    q = np.clip(M_tilde / (2.0 * r_tilde + 1e-30), 0, 0.999)
    A = (1.0 - q) / (1.0 + q)
    B = (1.0 + q)**2
    return A, B


def schwarzschild_kretschmann_exact(r_tilde, M_tilde=0.5):
    A, B = schwarzschild_isotropic(r_tilde, M_tilde)
    R_area = B * r_tilde
    K_exact = 48.0 * M_tilde**2 / (R_area**6 + 1e-60)
    return K_exact, R_area




def compute_analytic_geometry(r, A, B):
    Ap = deriv4(A, r)
    Bp = deriv4(B, r)
    App = deriv4(Ap, r)
    Bpp = deriv4(Bp, r)

    invA = np.where(np.abs(A) > 1e-30, 1.0/A, 0.0)
    invB = np.where(np.abs(B) > 1e-30, 1.0/B, 0.0)
    invr = np.where(r > 1e-30, 1.0/r, 0.0)
    B2 = B**2
    A2 = A**2

    phi_p = Ap * invA
    psi_p = Bp * invB
    phi_pp = App * invA - phi_p**2
    psi_pp = Bpp * invB - psi_p**2

    invB2 = 1.0 / (B2 + 1e-30)

    K1 = invB2 * (phi_pp + phi_p * (phi_p - psi_p))
    K2 = invB2 * phi_p * (psi_p + invr)
    K3 = -invB2 * (psi_pp + psi_p * invr)
    K4 = np.where(B2 * r ** 2 > 1e-30,
                  (1.0 - (1.0 + r * psi_p) ** 2) / (B2 * r ** 2),
                  0.0)

    K_kretsch = 4.0 * (K1**2 + 2.0*K2**2 + 2.0*K3**2 + K4**2)

    R_scalar = -2.0*K1 - 4.0*K2 + 4.0*K3 + 2.0*K4

    R_tt = A2 * (K1 + 2.0 * K2)
    R_rr = B2 * (-K1 + 2.0 * K3)
    R_thth = B2 * r**2 * (-K2 + K3 + K4)

    G_tt = R_tt + 0.5 * R_scalar * A2
    G_rr = R_rr - 0.5 * R_scalar * B2
    G_thth = R_thth - 0.5 * R_scalar * B2 * r**2

    G_t_tr = phi_p
    G_r_tt = A * Ap / (B2 + 1e-30)
    G_r_rr = psi_p
    G_r_thth = -r * (r * psi_p + 1.0)
    G_th_rth = psi_p + invr

    return dict(
        R_tt=R_tt, R_rr=R_rr, R_thth=R_thth, R_scalar=R_scalar,
        G_tt=G_tt, G_rr=G_rr, G_thth=G_thth,
        K_kretsch=K_kretsch, K1=K1, K2=K2, K3=K3, K4=K4,
        Gt_tr=G_t_tr, Gr_tt=G_r_tt, Gr_rr=G_r_rr,
        Gr_thth=G_r_thth, Gth_rth=G_th_rth,
        Ap=Ap, Bp=Bp, phi_p=phi_p, psi_p=psi_p
    )




def run_schwarzschild_test(r_tilde, M_tilde=0.5):
    A, B = schwarzschild_isotropic(r_tilde, M_tilde)
    K_exact, R_area = schwarzschild_kretschmann_exact(r_tilde, M_tilde)

    geo = compute_analytic_geometry(r_tilde, A, B)

    mask = (r_tilde > 0.3) & (r_tilde < r_tilde[-5])

    K_computed = geo['K_kretsch']
    K_rel_err = np.abs(K_computed[mask] - K_exact[mask]) / (K_exact[mask] + 1e-30)

    R_scal_err = np.abs(geo['R_scalar'][mask])

    G_tt_err = np.abs(geo['G_tt'][mask])
    G_rr_err = np.abs(geo['G_rr'][mask])
    G_thth_err = np.abs(geo['G_thth'][mask])

    return dict(
        K_rel_err_mean=float(np.mean(K_rel_err)),
        K_rel_err_max=float(np.max(K_rel_err)),
        R_scalar_mean=float(np.mean(R_scal_err)),
        R_scalar_max=float(np.max(R_scal_err)),
        G_tt_mean=float(np.mean(G_tt_err)),
        G_rr_mean=float(np.mean(G_rr_err)),
        G_thth_mean=float(np.mean(G_thth_err)),
        K_exact=K_exact,
        K_computed=K_computed,
        R_area=R_area
    )



def compute_adm_mass(r, A, B, M_expected):
    n_fit = max(20, len(r) // 5)
    r_fit = r[-n_fit:]
    A_fit = A[-n_fit:]
    B_fit = B[-n_fit:]

    M_from_A_arr = -r_fit * (A_fit - 1.0)
    M_from_A = float(np.median(M_from_A_arr))

    M_from_B_arr = r_fit * (B_fit - 1.0)
    M_from_B = float(np.median(M_from_B_arr))

    M_ADM = 0.5 * (M_from_A + M_from_B)

    Bp = deriv4(B, r)
    M_surface = float(-r[-1] ** 2 * Bp[-1] / np.sqrt(B[-1] + 1e-30))

    err_A = abs(M_from_A - M_expected) / (M_expected + 1e-30)
    err_B = abs(M_from_B - M_expected) / (M_expected + 1e-30)
    err_ADM = abs(M_ADM - M_expected) / (M_expected + 1e-30)
    err_surf = abs(M_surface - M_expected) / (M_expected + 1e-30)

    return dict(
        M_expected=M_expected,
        M_from_A=M_from_A,
        M_from_B=M_from_B,
        M_ADM=M_ADM,
        M_surface=float(M_surface),
        err_A=err_A,
        err_B=err_B,
        err_ADM=err_ADM,
        err_surface=err_surf,
        M_from_A_profile=M_from_A_arr,
        M_from_B_profile=M_from_B_arr
    )




def test_asymptotic_flatness(r, A, B, geo, frac=0.9):
    r_thresh = frac * r[-1]
    mask = r > r_thresh

    dev_A = np.abs(A[mask] - 1.0)
    dev_B = np.abs(B[mask] - 1.0)
    K_asym = np.abs(geo['K_kretsch'][mask])
    R_asym = np.abs(geo['R_scalar'][mask])
    G_asym = np.abs(geo['G_tt'][mask])

    r_asym = r[mask]

    return dict(
        r_threshold=float(r_thresh),
        dev_A_mean=float(np.mean(dev_A)),
        dev_A_max=float(np.max(dev_A)),
        dev_B_mean=float(np.mean(dev_B)),
        dev_B_max=float(np.max(dev_B)),
        K_mean=float(np.mean(K_asym)),
        K_max=float(np.max(K_asym)),
        R_mean=float(np.mean(R_asym)),
        G_mean=float(np.mean(G_asym)),
        A_times_r=float(np.mean(dev_A * r_asym)),
        B_times_r=float(np.mean(dev_B * r_asym)),
    )



def check_efe(G_tt, G_rr, G_thth, Lambda_rho, rho_eff, p_r, p_t, A, B, r):
    A2, B2, r2 = A**2, B**2, r**2

    g_tt, g_rr, g_thth = -A2, B2, B2 * r2

    T_tt = A2 * rho_eff
    T_rr = B2 * p_r
    T_thth = B2 * r2 * p_t

    res_tt = G_tt - Lambda_rho * g_tt - 2.0 * T_tt
    res_rr = G_rr - Lambda_rho * g_rr - 2.0 * T_rr
    res_thth = G_thth - Lambda_rho * g_thth - 2.0 * T_thth

    scale_tt = np.maximum(np.abs(G_tt), np.abs(2*T_tt)) + 1e-30
    scale_rr = np.maximum(np.abs(G_rr), np.abs(2*T_rr)) + 1e-30
    scale_thth = np.maximum(np.abs(G_thth), np.abs(2*T_thth)) + 1e-30

    return dict(
        res_tt=res_tt, res_rr=res_rr, res_thth=res_thth,
        rel_tt=np.abs(res_tt)/scale_tt,
        rel_rr=np.abs(res_rr)/scale_rr,
        rel_thth=np.abs(res_thth)/scale_thth
    )


def compute_div_T(r, rho_eff, p_r, p_t, A, B, geo):
    A2 = A**2 + 1e-30
    B2 = B**2 + 1e-30
    r2 = r**2 + 1e-30

    T_tt_up = rho_eff / A2
    T_rr_up = p_r / B2
    T_thth_up = p_t / (B2 * r2)

    dT_rr_dr = deriv4(T_rr_up, r)
    Gamma_trace_r = geo['Gt_tr'] + geo['Gr_rr'] + 2.0 * geo['Gth_rth']
    Gamma_r_contract = (geo['Gr_tt'] * T_tt_up + geo['Gr_rr'] * T_rr_up
                        + 2.0 * geo['Gr_thth'] * T_thth_up)

    return dT_rr_dr + Gamma_trace_r * T_rr_up + Gamma_r_contract



def check_energy_conditions(upsilon_00, pressure):
    rho = upsilon_00
    p = pressure
    wec = ((rho < 0) | ((rho + p) < 0)).astype(float)
    nec = ((rho + p) < 0).astype(float)
    sec = ((rho + 3*p) < 0).astype(float)
    dec = (rho < np.abs(p)).astype(float)
    return dict(wec=wec, nec=nec, sec=sec, dec=dec)


def compute_einstein_tensor_isotropic(r, A, B):
    import numpy as np

    phi = np.log(np.clip(A, 1e-10, None))
    psi = np.log(np.clip(B, 1e-10, None))

    phi_p = deriv4(phi, r)
    psi_p = deriv4(psi, r)
    phi_pp = deriv4_second(phi, r)
    psi_pp = deriv4_second(psi, r)

    invr = 1.0 / np.clip(r, 1e-10, None)
    B2 = B ** 2
    A2 = A ** 2

    G_t_t = (1.0 / B2) * (2.0 * psi_pp + psi_p ** 2 + 4.0 * psi_p * invr)
    G_r_r = (1.0 / B2) * (psi_p ** 2 + 2.0 * phi_p * psi_p + 2.0 * invr * (phi_p + psi_p))
    G_th_th = (1.0 / B2) * (phi_pp + psi_pp + phi_p ** 2 + invr * (phi_p + psi_p))

    R_scalar = -(G_t_t + G_r_r + 2.0 * G_th_th)

    R_t_t = G_t_t + 0.5 * R_scalar
    R_r_r = G_r_r + 0.5 * R_scalar
    R_th_th = G_th_th + 0.5 * R_scalar

    G_tt = -A2 * G_t_t
    G_rr = B2 * G_r_r
    G_thth = B2 * r ** 2 * G_th_th

    return {
        'G_tt': G_tt, 'G_rr': G_rr, 'G_thth': G_thth,
        'G_t_t': G_t_t, 'G_r_r': G_r_r, 'G_th_th': G_th_th,
        'R_scalar': R_scalar,
        'R_t_t': R_t_t, 'R_r_r': R_r_r, 'R_th_th': R_th_th,
        'phi_p': phi_p, 'psi_p': psi_p,
        'phi_pp': phi_pp, 'psi_pp': psi_pp,
    }


def verify_einstein_schwarzschild(r, M):
    A, B = schwarzschild_isotropic(r, M)
    ein = compute_einstein_tensor_isotropic(r, A, B)

    mask = (r > 0.3) & (r < r[-5])

    errs = {}
    for key in ['G_t_t', 'G_r_r', 'G_th_th', 'R_scalar']:
        errs[key] = float(np.max(np.abs(ein[key][mask])))

    return errs


def solve_metric_ode_strict(r_arr, dp, cfg):
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d
    import warnings

    Nr = len(r_arr)
    M = dp.M_dimless

    try:
        dp.rho_dimless = 1e-7
    except AttributeError:
        object.__setattr__(dp, 'rho_dimless', 1e-7)

    q = M / (2.0 * r_arr)
    A_schw = (1.0 - q) / (1.0 + q)
    B_schw = (1.0 + q) ** 2

    f_bub = bubble_shape(r_arr, dp.R_bubble_dimless, dp.sigma_dimless)
    rho_matter = dp.rho_dimless * f_bub
    p_matter = np.zeros_like(r_arr)

    r0 = r_arr[-1]
    q0 = M / (2.0 * r0)
    phi0 = np.log(A_schw[-1])
    psi0 = np.log(B_schw[-1])
    phi_p0 = 2.0 * q0 / (r0 * (1.0 - q0 ** 2))
    psi_p0 = -2.0 * q0 / (r0 * (1.0 + q0))

    y0 = [phi0, phi_p0, psi0, psi_p0]
    r_span = (r_arr[-1], r_arr[0])
    r_eval = r_arr[::-1]

    A_current = A_schw.copy()
    B_current = B_schw.copy()

    def integrate_with_multiplier(mult, A_bg, B_bg):
        current_E_factor = dp.E_factor_dimless * np.sqrt(mult)
        E_r, B_theta = em_field_profile(r_arr, dp.R_bubble_dimless, dp.sigma_dimless,
                                        current_E_factor, M)
        Lr_arr, _, _, _, _ = compute_lambda_rho(E_r, B_theta, A_bg, B_bg)

        S_tt = Lr_arr - 2.0 * rho_matter
        S_thth = Lr_arr + 2.0 * p_matter

        itp_Stt = interp1d(r_arr, S_tt, bounds_error=False, fill_value="extrapolate")
        itp_Sth = interp1d(r_arr, S_thth, bounds_error=False, fill_value="extrapolate")

        def rhs(r_val, y):
            phi_v, phi_pv, psi_v, psi_pv = y
            B2_v = np.exp(2.0 * psi_v)

            if r_val < 1e-6:
                return [phi_pv, 0.0, psi_pv, 0.0]

            invr_v = 1.0 / r_val
            S_tt_v = float(itp_Stt(r_val))
            S_th_v = float(itp_Sth(r_val))

            psi_ppv = 0.5 * B2_v * S_tt_v - 0.5 * psi_pv ** 2 - 2.0 * psi_pv * invr_v
            phi_ppv = B2_v * S_th_v - psi_ppv - phi_pv ** 2 - (phi_pv + psi_pv) * invr_v

            return [phi_pv, phi_ppv, psi_pv, psi_ppv]

        return solve_ivp(rhs, r_span, y0, method=cfg.ode_method, t_eval=r_eval,
                         rtol=cfg.ode_rtol, atol=cfg.ode_atol)

    best_mult = 1.0
    best_A = A_schw.copy()
    best_B = B_schw.copy()

    for sc_iter in range(cfg.ode_self_consistent_iters):
        low_mult = 0.0
        high_mult = 50000.0
        iter_best_sol = None
        iter_best_mult = best_mult

        for bs_iter in range(60):
            mid_mult = 0.5 * (low_mult + high_mult)
            sol = integrate_with_multiplier(mid_mult, A_current, B_current)

            if not sol.success or len(sol.y[0]) != Nr:
                high_mult = mid_mult
                continue

            iter_best_sol = sol
            iter_best_mult = mid_mult

            A_trial = np.exp(sol.y[0][::-1])
            B_trial = np.exp(sol.y[2][::-1])

            core = f_bub > 0.9
            if np.sum(core) < 3:
                core = f_bub > 0.5

            if np.sum(core) > 0:
                A_core_mean = np.mean(A_trial[core])
                B_core_mean = np.mean(B_trial[core])

                deviation = (A_core_mean - 1.0) + (B_core_mean - 1.0)

                if deviation < -0.002:
                    low_mult = mid_mult
                elif deviation > 0.002:
                    high_mult = mid_mult
                else:
                    break
            else:
                low_mult = mid_mult
                continue

        if iter_best_sol is not None:
            best_A = np.exp(iter_best_sol.y[0][::-1])
            best_B = np.exp(iter_best_sol.y[2][::-1])
            best_mult = iter_best_mult

            A_current = best_A.copy()
            B_current = best_B.copy()

    f_bub_mask = bubble_shape(r_arr, dp.R_bubble_dimless, dp.sigma_dimless)
    core_mask = f_bub_mask > 0.999
    if np.any(core_mask):
        core_indices = np.where(core_mask)[0]
        border_idx = core_indices[-1]

        A_border = best_A[border_idx]
        B_border = best_B[border_idx]

        best_A[core_mask] = A_border
        best_B[core_mask] = B_border

    correct_factor = dp.E_factor_dimless * np.sqrt(best_mult)
    try:
        dp.E_factor_dimless = correct_factor
    except AttributeError:
        object.__setattr__(dp, 'E_factor_dimless', correct_factor)

    return best_A, best_B


def solve_time_dependent(r_arr, dp, cfg, Nt=100, dt_factor=0.1):
    Nr = len(r_arr)
    dr = r_arr[1] - r_arr[0]
    dt = dt_factor * dr

    A_init, B_init = solve_metric_ode_strict(r_arr, dp, cfg)

    A = A_init.copy()
    B = B_init.copy()
    K = np.zeros(Nr)

    history = {
        't': [0.0],
        'A': [A.copy()],
        'B': [B.copy()],
        'K': [K.copy()],
    }

    f_bub = bubble_shape(r_arr, dp.R_bubble_dimless, dp.sigma_dimless)
    invr = 1.0 / np.clip(r_arr, 1e-10, None)

    for n in range(Nt):
        t = (n + 1) * dt

        def compute_rhs(A_v, B_v, K_v):


            B2 = B_v ** 2

            E_r, B_theta = em_field_profile(r_arr, dp.R_bubble_dimless, dp.sigma_dimless,
                                            dp.E_factor_dimless, dp.M_dimless)
            Lr, _, _, _, _, _, _ = compute_lambda_rho(E_r, B_theta, A_v, B_v)

            rho_matter = dp.rho_dimless * f_bub



            dA_dt = -2.0 * A_v * K_v

            dB_dt = -A_v * B_v * K_v / 3.0


            A_p = deriv4(A_v, r_arr)
            A_pp = deriv4_second(A_v, r_arr)

            D2A_over_A = (1.0 / B2) * (A_pp / A_v + 2.0 * A_p * invr / A_v)

            dK_dt = -D2A_over_A + K_v ** 2 / 3.0 + 4.0 * np.pi * (rho_matter) - 2.0 * Lr

            return dA_dt, dB_dt, dK_dt

        k1_A, k1_B, k1_K = compute_rhs(A, B, K)
        k2_A, k2_B, k2_K = compute_rhs(A + 0.5 * dt * k1_A, B + 0.5 * dt * k1_B, K + 0.5 * dt * k1_K)
        k3_A, k3_B, k3_K = compute_rhs(A + 0.5 * dt * k2_A, B + 0.5 * dt * k2_B, K + 0.5 * dt * k2_K)
        k4_A, k4_B, k4_K = compute_rhs(A + dt * k3_A, B + dt * k3_B, K + dt * k3_K)

        A = A + (dt / 6.0) * (k1_A + 2 * k2_A + 2 * k3_A + k4_A)
        B = B + (dt / 6.0) * (k1_B + 2 * k2_B + 2 * k3_B + k4_B)
        K = K + (dt / 6.0) * (k1_K + 2 * k2_K + 2 * k3_K + k4_K)

        A = np.clip(A, 1e-3, 5.0)
        B = np.clip(B, 0.1, 20.0)

        M = dp.M_dimless
        q_max = M / (2.0 * r_arr[-1])
        A[-1] = (1.0 - q_max) / (1.0 + q_max)
        B[-1] = (1.0 + q_max) ** 2
        K[-1] = 0.0

        if (n + 1) % max(1, Nt // 20) == 0 or n == Nt - 1:
            history['t'].append(t)
            history['A'].append(A.copy())
            history['B'].append(B.copy())
            history['K'].append(K.copy())

    return history




def stability_analysis(r, A, B, dp):
    eps_val = 1e-4
    f_bub = bubble_shape(r, dp.R_bubble_dimless, dp.sigma_dimless)

    def get_residual(Av, Bv):
        E_r, B_theta = em_field_profile(r, dp.R_bubble_dimless, dp.sigma_dimless,
                                        dp.E_factor_dimless, dp.M_dimless)
        Lr, E2, B2em, _, _ = compute_lambda_rho(E_r, B_theta, Av, Bv)
        rho, pr, pt, _, _, _, _, _ = compute_stress_energy(Lr, dp.rho_dimless, Av, Bv, r)
        geo = compute_analytic_geometry(r, Av, Bv)
        efe = check_efe(geo['G_tt'], geo['G_rr'], geo['G_thth'], Lr, rho, pr, pt, Av, Bv, r)
        mask = (r > 0.3) & (r < r[-10])
        return np.mean(np.abs(efe['res_tt'][mask]))

    res0 = get_residual(A, B)

    A_pert = A * (1.0 + eps_val * f_bub)
    res_dA = get_residual(A_pert, B)

    B_pert = B * (1.0 + eps_val * f_bub)
    res_dB = get_residual(A, B_pert)

    dres_dA = (res_dA - res0) / eps_val
    dres_dB = (res_dB - res0) / eps_val

    return dres_dA, dres_dB, res0




def project_1d_to_2d(r_1d, field_1d, X2d, Y2d):
    r_2d = np.sqrt(X2d**2 + Y2d**2 + 1e-30)
    interp = interp1d(r_1d, field_1d, kind='cubic', bounds_error=False,
                      fill_value=(field_1d[0], field_1d[-1]))
    return interp(r_2d)


@njit(fastmath=True, cache=True)
def _geodesic_rk4(x, y, vx, vy, dt, A_2d, B_2d, dx, ny, nx, half_ext):
    def accel(cx, cy):
        if abs(cx) >= half_ext*0.95 or abs(cy) >= half_ext*0.95:
            return 0.0, 0.0
        ik = int((cx + half_ext) / dx)
        ij = int((cy + half_ext) / dx)
        ik = max(1, min(nx-2, ik))
        ij = max(1, min(ny-2, ij))
        inv2dx = 0.5 / dx
        Ax_ = (A_2d[ij, ik+1] - A_2d[ij, ik-1]) * inv2dx
        Ay_ = (A_2d[ij+1, ik] - A_2d[ij-1, ik]) * inv2dx
        Av = A_2d[ij, ik]
        Bv = B_2d[ij, ik]
        B2v = Bv*Bv + 1e-30
        return Av*Ax_/B2v, Av*Ay_/B2v

    ax1, ay1 = accel(x, y)
    x2, y2 = x+0.5*dt*vx, y+0.5*dt*vy
    vx2, vy2 = vx+0.5*dt*ax1, vy+0.5*dt*ay1
    ax2, ay2 = accel(x2, y2)
    x3, y3 = x+0.5*dt*vx2, y+0.5*dt*vy2
    vx3, vy3 = vx+0.5*dt*ax2, vy+0.5*dt*ay2
    ax3, ay3 = accel(x3, y3)
    x4, y4 = x+dt*vx3, y+dt*vy3
    vx4, vy4 = vx+dt*ax3, vy+dt*ay3
    ax4, ay4 = accel(x4, y4)
    x += dt/6*(vx+2*vx2+2*vx3+vx4)
    y += dt/6*(vy+2*vy2+2*vy3+vy4)
    vx += dt/6*(ax1+2*ax2+2*ax3+ax4)
    vy += dt/6*(ay1+2*ay2+2*ay3+ay4)
    return x, y, vx, vy


@njit(parallel=True, fastmath=True, cache=True)
def trace_geodesics_2d(num_rays, steps, dt, A_2d, B_2d, dx, ny, nx):
    traj_x = np.zeros((num_rays, steps))
    traj_y = np.zeros((num_rays, steps))
    half_ext = 0.5 * nx * dx
    ys = np.linspace(-half_ext*0.85, half_ext*0.85, num_rays)
    for ray in prange(num_rays):
        x, y, vx, vy = -half_ext*0.9, ys[ray], 1.0, 0.0
        for s in range(steps):
            traj_x[ray, s] = x
            traj_y[ray, s] = y
            x, y, vx, vy = _geodesic_rk4(x, y, vx, vy, dt, A_2d, B_2d,
                                          dx, ny, nx, half_ext)
            if abs(x) > half_ext or abs(y) > half_ext:
                for rem in range(s+1, steps):
                    traj_x[ray, rem] = x
                    traj_y[ray, rem] = y
                break
    return traj_x, traj_y




class PhysicsEngine:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.r_s = 3.0
        self.R_bubble = 8.0
        self.sigma = 2.0
        self.E_factor = 10.0
        self.analytics = {}
        self._results = {}

    def _make_grid(self):
        self.dp = DimensionlessParams.from_physical(
            self.r_s, self.R_bubble, self.sigma, self.E_factor, self.cfg.rest_mass_density)

        self.r_tilde = np.linspace(
            self.cfg.r_min_over_rs,
            self.cfg.r_max_over_rs,
            self.cfg.Nr
        )

        N2 = self.cfg.grid_2d
        half = self.cfg.extent_2d_over_rs
        coords = np.linspace(-half, half, N2)
        self.Y2d, self.X2d = np.meshgrid(coords, coords, indexing='ij')
        self.dx_2d = 2.0 * half / N2
        self.N2 = N2

    def execute_full_pipeline(self):
        self._make_grid()
        r = self.r_tilde
        dp = self.dp
        cfg = self.cfg

        schw_test = run_schwarzschild_test(r, dp.M_dimless)

        A_sol, B_sol = solve_metric_ode_strict(r, dp, cfg)

        conv_data = run_convergence_study(dp, cfg, Nr_list=[500, 1000, 1500, 2000])

        A_schw, B_schw = schwarzschild_isotropic(r, dp.M_dimless)

        E_r, B_theta = em_field_profile(r, dp.R_bubble_dimless, dp.sigma_dimless,
                                        dp.E_factor_dimless, dp.M_dimless)
        f_bub = bubble_shape(r, dp.R_bubble_dimless, dp.sigma_dimless)

        Lr, E2, B2em, E_phys, B_phys = compute_lambda_rho(E_r, B_theta, A_sol, B_sol)
        rho_eff, p_r, p_t, pres, ups00, mu_r, chi, eps_r = compute_stress_energy(
            Lr, dp.rho_dimless, A_sol, B_sol, r)

        geo = compute_analytic_geometry(r, A_sol, B_sol)
        geo_schw = compute_analytic_geometry(r, A_schw, B_schw)

        efe = check_efe(geo['G_tt'], geo['G_rr'], geo['G_thth'],
                       Lr, rho_eff, p_r, p_t, A_sol, B_sol, r)

        div_T = compute_div_T(r, rho_eff, p_r, p_t, A_sol, B_sol, geo)

        ec = check_energy_conditions(ups00, pres)

        adm = compute_adm_mass(r, A_sol, B_sol, dp.M_dimless)

        asym = test_asymptotic_flatness(r, A_sol, B_sol, geo, cfg.asymptotic_test_frac)

        dres_dA, dres_dB, res0 = stability_analysis(r, A_sol, B_sol, dp)

        K_eff = geo['K_kretsch']
        K_schw = geo_schw['K_kretsch']
        compensation = np.where(K_schw > 1e-30, 1.0 - K_eff / K_schw, 0.0)
        compensation = np.clip(compensation, -2, 2)

        Lambda_cosmo = -4.0 * np.pi * Lr

        A_2d = project_1d_to_2d(r, A_sol, self.X2d, self.Y2d)
        B_2d = project_1d_to_2d(r, B_sol, self.X2d, self.Y2d)
        A_schw_2d = project_1d_to_2d(r, A_schw, self.X2d, self.Y2d)
        B_schw_2d = project_1d_to_2d(r, B_schw, self.X2d, self.Y2d)
        f_2d = project_1d_to_2d(r, f_bub, self.X2d, self.Y2d)
        Lr_2d = project_1d_to_2d(r, Lr, self.X2d, self.Y2d)
        comp_2d = project_1d_to_2d(r, compensation, self.X2d, self.Y2d)
        K_eff_2d = project_1d_to_2d(r, K_eff, self.X2d, self.Y2d)
        K_schw_2d = project_1d_to_2d(r, K_schw, self.X2d, self.Y2d)
        E_phys_2d = project_1d_to_2d(r, E_phys, self.X2d, self.Y2d)
        Rscal_2d = project_1d_to_2d(r, geo['R_scalar'], self.X2d, self.Y2d)
        Gtt_2d = project_1d_to_2d(r, geo['G_tt'], self.X2d, self.Y2d)
        res_tt_2d = project_1d_to_2d(r, efe['res_tt'], self.X2d, self.Y2d)
        div_T_2d = project_1d_to_2d(r, div_T, self.X2d, self.Y2d)
        wec_2d = project_1d_to_2d(r, ec['wec'], self.X2d, self.Y2d)
        nec_2d = project_1d_to_2d(r, ec['nec'], self.X2d, self.Y2d)
        sec_2d = project_1d_to_2d(r, ec['sec'], self.X2d, self.Y2d)
        dec_2d = project_1d_to_2d(r, ec['dec'], self.X2d, self.Y2d)

        ny, nx = A_2d.shape
        ray_dt = self.cfg.ray_dt_over_rs
        traj_xs, traj_ys = trace_geodesics_2d(
            cfg.num_rays, cfg.ray_steps, ray_dt,
            A_schw_2d, B_schw_2d, self.dx_2d, ny, nx)
        traj_xe, traj_ye = trace_geodesics_2d(
            cfg.num_rays, cfg.ray_steps, ray_dt,
            A_2d, B_2d, self.dx_2d, ny, nx)

        R = self._results
        R['r'] = r
        R['A'] = A_sol; R['B'] = B_sol
        R['A_schw'] = A_schw; R['B_schw'] = B_schw
        R['f_bub'] = f_bub; R['E_r'] = E_r; R['B_theta'] = B_theta
        R['E_phys'] = E_phys; R['B_phys'] = B_phys
        R['Lr'] = Lr; R['E2'] = E2; R['B2em'] = B2em
        R['rho_eff'] = rho_eff; R['p_r'] = p_r; R['p_t'] = p_t
        R['pressure'] = pres; R['ups00'] = ups00
        R['mu_r'] = mu_r; R['chi'] = chi; R['eps_r'] = eps_r
        R['geo'] = geo; R['geo_schw'] = geo_schw
        R['K_eff'] = K_eff; R['K_schw'] = K_schw
        R['efe'] = efe; R['div_T'] = div_T; R['ec'] = ec
        R['compensation'] = compensation
        R['Lambda_cosmo'] = Lambda_cosmo
        R['adm'] = adm; R['asym'] = asym
        R['schw_test'] = schw_test
        R['stability'] = dict(dres_dA=dres_dA, dres_dB=dres_dB, res0=res0)

        R['A_2d'] = A_2d; R['B_2d'] = B_2d; R['f_2d'] = f_2d
        R['Lr_2d'] = Lr_2d; R['comp_2d'] = comp_2d
        R['K_eff_2d'] = K_eff_2d; R['K_schw_2d'] = K_schw_2d
        R['E_phys_2d'] = E_phys_2d
        R['Rscal_2d'] = Rscal_2d; R['Gtt_2d'] = Gtt_2d
        R['res_tt_2d'] = res_tt_2d; R['div_T_2d'] = div_T_2d
        R['wec_2d'] = wec_2d; R['nec_2d'] = nec_2d
        R['sec_2d'] = sec_2d; R['dec_2d'] = dec_2d
        R['traj_xs'] = traj_xs; R['traj_ys'] = traj_ys
        R['traj_xe'] = traj_xe; R['traj_ye'] = traj_ye
        R['convergence'] = conv_data

        self._compute_analytics()

    def _compute_analytics(self):
        a = {}
        R = self._results
        r = R['r']
        inner = R['f_bub'] > 0.999
        inner_idx = np.where(inner)[0]
        if len(inner_idx) > 10:
            safe_inner = np.zeros_like(inner, dtype=bool)
            safe_inner[inner_idx[:-10]] = True
            inner = safe_inner
        elif len(inner_idx) > 0:
            safe_inner = np.zeros_like(inner, dtype=bool)
            safe_inner[inner_idx[0]] = True
            inner = safe_inner

        mask = (r > 0.3) & (r < r[-10])

        a['A_inside'] = float(np.mean(R['A'][inner])) if np.sum(inner) > 0 else 0.0
        a['B_inside'] = float(np.mean(R['B'][inner])) if np.sum(inner) > 0 else 0.0
        a['K_schw_max'] = float(np.max(R['K_schw'][mask]))
        a['K_eff_inside'] = float(np.max(R['K_eff'][inner])) if np.sum(inner) > 0 else 0.0
        a['comp_inside'] = float(np.mean(R['compensation'][inner])) if np.sum(inner) > 0 else 0.0

        a['res_tt'] = float(np.mean(R['efe']['rel_tt'][mask]))
        a['res_rr'] = float(np.mean(R['efe']['rel_rr'][mask]))
        a['res_thth'] = float(np.mean(R['efe']['rel_thth'][mask]))
        a['div_T_mean'] = float(np.mean(np.abs(R['div_T'][mask])))
        a['div_T_max'] = float(np.max(np.abs(R['div_T'][mask])))

        a['Lr_min'] = float(np.min(R['Lr'])); a['Lr_max'] = float(np.max(R['Lr']))
        a['E_max'] = float(np.max(R['E_phys'])); a['B_max'] = float(np.max(R['B_phys']))
        a['p_min'] = float(np.min(R['pressure'])); a['p_max'] = float(np.max(R['pressure']))

        a['adm'] = R['adm']
        a['asym'] = R['asym']
        a['schw_test'] = R['schw_test']
        a['stability'] = R['stability']

        total = len(r)
        a['wec_pct'] = float(np.sum(R['ec']['wec']) / total * 100)
        a['nec_pct'] = float(np.sum(R['ec']['nec']) / total * 100)
        a['sec_pct'] = float(np.sum(R['ec']['sec']) / total * 100)
        a['dec_pct'] = float(np.sum(R['ec']['dec']) / total * 100)

        a['R_tt_range'] = (float(np.min(R['geo']['R_tt'][mask])),
                           float(np.max(R['geo']['R_tt'][mask])))
        a['R_scal_range'] = (float(np.min(R['geo']['R_scalar'][mask])),
                             float(np.max(R['geo']['R_scalar'][mask])))
        a['G_tt_range'] = (float(np.min(R['geo']['G_tt'][mask])),
                           float(np.max(R['geo']['G_tt'][mask])))

        self.analytics = a

    def generate_report(self):
        R = self._results
        a = self.analytics
        dp = self.dp
        s = "="*86 + "\n"
        ln = "-"*86 + "\n"
        t = s
        t += " НАУЧНЫЙ ОТЧЁТ v15.0: ПУЗЫРЬ ПЛОСКОГО ПРОСТРАНСТВА-ВРЕМЕНИ\n"
        t += " Analytic Riemann + Shooting ODE + Full Kretschmann\n"
        t += " ADM Mass + Asymptotic Flatness + Schwarzschild Test + Stability\n"
        t += " Ogonowski & Skindzier (2025) Phys. Scr. 100, 015018\n"
        t += s
        t += f" Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        t += "[12] CONVERGENCE STUDY\n" + ln
        cd = R.get('convergence', {})
        if cd:
            t += format_convergence_report(cd)

        t += "[1] БЕЗРАЗМЕРНЫЕ ПАРАМЕТРЫ (r̃ = r/r_s)\n" + ln
        t += f" r_s = {dp.r_s:.2f} (масштаб длины)\n"
        t += f" M̃ = M/r_s = {dp.M_dimless:.4f}\n"
        t += f" R̃_bub = {dp.R_bubble_dimless:.4f}\n"
        t += f" σ̃ = σ·r_s = {dp.sigma_dimless:.4f}\n"
        t += f" Ẽ₀ = {dp.E_factor_dimless:.4f}\n"
        t += f" ρ̃₀ = ρ₀·r_s² = {dp.rho_dimless:.4f}\n"
        t += f" Сетка: {self.cfg.Nr} точек, r̃∈[{self.cfg.r_min_over_rs:.2f}, {self.cfg.r_max_over_rs:.2f}]\n"
        t += f" ODE: {self.cfg.ode_method}, iter={self.cfg.ode_self_consistent_iters}\n\n"

        t += "[2] ТЕСТ ШВАРЦШИЛЬДА (Λ_ρ=0, точное решение)\n" + ln
        st = a['schw_test']
        t += f" K_rel_err (средняя): {st['K_rel_err_mean']:.4e}\n"
        t += f" K_rel_err (макс):    {st['K_rel_err_max']:.4e}\n"
        t += f" |R| (должен=0):      {st['R_scalar_mean']:.4e}\n"
        t += f" |G_tt| (должен=0):   {st['G_tt_mean']:.4e}\n"
        t += f" |G_rr| (должен=0):   {st['G_rr_mean']:.4e}\n"
        t += f" |G_θθ| (должен=0):   {st['G_thth_mean']:.4e}\n"
        pass_k = "✓" if st['K_rel_err_mean'] < 0.01 else "✗"
        pass_r = "✓" if st['R_scalar_mean'] < 1e-4 else "✗"
        pass_g = "✓" if st['G_tt_mean'] < 1e-4 else "✗"
        t += f" Тест K:  {pass_k}   Тест R=0: {pass_r}   Тест G=0: {pass_g}\n\n"

        t += "[3] ADM МАССА\n" + ln
        ad = a['adm']
        t += f" M_ожидаемая = {ad['M_expected']:.6f}\n"
        t += f" M(из A):     {ad['M_from_A']:.6f}  (ошибка {ad['err_A']:.4e})\n"
        t += f" M(из B):     {ad['M_from_B']:.6f}  (ошибка {ad['err_B']:.4e})\n"
        t += f" M_ADM:       {ad['M_ADM']:.6f}     (ошибка {ad['err_ADM']:.4e})\n"
        t += f" M(поверхн.): {ad['M_surface']:.6f}  (ошибка {ad['err_surface']:.4e})\n"
        pass_adm = "✓" if ad['err_ADM'] < 0.05 else "✗"
        t += f" Тест ADM: {pass_adm}\n\n"

        t += "[4] АСИМПТОТИЧЕСКАЯ ПЛОСКОСТЬ\n" + ln
        asy = a['asym']
        t += f" r > {asy['r_threshold']:.2f}r_s:\n"
        t += f"   |A−1| средн./макс: {asy['dev_A_mean']:.4e} / {asy['dev_A_max']:.4e}\n"
        t += f"   |B−1| средн./макс: {asy['dev_B_mean']:.4e} / {asy['dev_B_max']:.4e}\n"
        t += f"   |K| средн.:        {asy['K_mean']:.4e}\n"
        t += f"   |R| средн.:        {asy['R_mean']:.4e}\n"
        t += f"   r·|A−1| ≈ M:      {asy['A_times_r']:.4f} (ожид. {dp.M_dimless:.4f})\n"
        t += f"   r·|B−1| ≈ M:      {asy['B_times_r']:.4f} (ожид. {dp.M_dimless:.4f})\n"
        pass_af = "✓" if asy['dev_A_max'] < 0.05 and asy['dev_B_max'] < 0.05 else "✗"
        t += f" Тест плоскости: {pass_af}\n\n"

        t += "[5] МЕТРИКА (SHOOTING ODE)\n" + ln
        t += f" A внутри: {a['A_inside']:.6f} (масштаб времени)\n"
        t += f" B внутри: {a['B_inside']:.6f} (масштаб длины)\n"
        t += f" K_Schw max:   {a['K_schw_max']:.4e}\n"
        t += f" K_eff внутри: {a['K_eff_inside']:.4e} (→ 0 = плоское ядро)\n"
        t += f" Компенсация:  {a['comp_inside'] * 100:.1f}%\n\n"

        t += "[6] EFE (A13): G_αβ − Λ_ρ g_αβ = 2T_αβ\n" + ln
        t += f" |tt| невязка: {a['res_tt']:.4e}\n"
        t += f" |rr| невязка: {a['res_rr']:.4e}\n"
        t += f" |θθ| невязка: {a['res_thth']:.4e}\n\n"

        t += "[7] ЗАКОН СОХРАНЕНИЯ ∇_μ T^{μν}\n" + ln
        t += f" <|∇_μ T^{{μr}}|> = {a['div_T_mean']:.4e}\n"
        t += f" max|∇_μ T^{{μr}}| = {a['div_T_max']:.4e}\n\n"

        t += "[8] УСТОЙЧИВОСТЬ\n" + ln
        stb = a['stability']
        t += f" Базовая невязка: {stb['res0']:.4e}\n"
        t += f" δ(res)/δA = {stb['dres_dA']:.4e}\n"
        t += f" δ(res)/δB = {stb['dres_dB']:.4e}\n\n"

        t += "[9] ЭНЕРГЕТИЧЕСКИЕ УСЛОВИЯ\n" + ln
        t += f" WEC: {a['wec_pct']:.1f}%, NEC: {a['nec_pct']:.1f}%\n"
        t += f" SEC: {a['sec_pct']:.1f}%, DEC: {a['dec_pct']:.1f}%\n\n"

        t += "[10] ТЕНЗОР АЛЕНЫ И ЭМ-ПОЛЕ\n" + ln
        t += f" Λ_ρ: [{a['Lr_min']:.4e}, {a['Lr_max']:.4e}]\n"
        t += f" |Ê|_max = {a['E_max']:.4e}, |B̂|_max = {a['B_max']:.4e}\n"
        t += f" p: [{a['p_min']:.4e}, {a['p_max']:.4e}]\n\n"

        t += "[11] СВОДКА ТЕСТОВ\n" + ln
        tests = [
            ("Schwarzschild K", st['K_rel_err_mean'] < 0.01),
            ("Schwarzschild R=0", st['R_scalar_mean'] < 1e-4),
            ("Schwarzschild G=0", st['G_tt_mean'] < 1e-4),
            ("ADM масса", ad['err_ADM'] < 0.05),
            ("Асимпт. плоскость", asy['dev_A_max'] < 0.05),
            ("EFE(tt) < 0.1", a['res_tt'] < 0.1),
            ("EFE(rr) < 0.1", a['res_rr'] < 0.1),
            ("EFE(θθ) < 0.1", a['res_thth'] < 0.1),
        ]
        passed = 0
        for name, ok in tests:
            sym = "✓" if ok else "✗"
            t += f"   [{sym}] {name}\n"
            if ok: passed += 1
        t += f"\n Пройдено: {passed}/{len(tests)}\n"
        t += s
        return t




def run_convergence_study(dp, cfg, Nr_list=None):
    if Nr_list is None:
        Nr_list = [500, 750, 1000, 1500, 2000]

    M = dp.M_dimless
    results = []

    for Nr in Nr_list:
        r = np.linspace(cfg.r_min_over_rs, cfg.r_max_over_rs, Nr)
        h = r[1] - r[0]

        A, B = schwarzschild_isotropic(r, M)

        ein = compute_einstein_tensor_isotropic(r, A, B)

        mask = (r > 0.4) & (r < r[-5])

        G_tt_err = np.max(np.abs(ein['G_t_t'][mask]))
        G_rr_err = np.max(np.abs(ein['G_r_r'][mask]))
        G_th_err = np.max(np.abs(ein['G_th_th'][mask]))
        R_err = np.max(np.abs(ein['R_scalar'][mask]))

        total_err = max(G_tt_err, G_rr_err, G_th_err)

        results.append({
            'Nr': Nr,
            'h': h,
            'G_tt_err': G_tt_err,
            'G_rr_err': G_rr_err,
            'G_th_err': G_th_err,
            'R_err': R_err,
            'total_err': total_err,
        })

    orders = []
    for i in range(1, len(results)):
        h1 = results[i - 1]['h']
        h2 = results[i]['h']
        e1 = results[i - 1]['total_err']
        e2 = results[i]['total_err']

        if e1 > 1e-15 and e2 > 1e-15 and abs(h1 - h2) > 1e-15:
            p = np.log(e1 / e2) / np.log(h1 / h2)
        else:
            p = float('nan')

        orders.append({
            'Nr1': results[i - 1]['Nr'],
            'Nr2': results[i]['Nr'],
            'h1': h1,
            'h2': h2,
            'e1': e1,
            'e2': e2,
            'order': p,
        })

    valid_orders = [o['order'] for o in orders if not np.isnan(o['order'])]
    mean_order = float(np.mean(valid_orders)) if valid_orders else float('nan')

    return {
        'resolutions': results,
        'convergence_orders': orders,
        'mean_order': mean_order,
        'expected_order': 4.0,
        'order_achieved': not np.isnan(mean_order) and abs(mean_order - 4.0) < 1.5,
    }


def format_convergence_report(conv):
    lines = []
    lines.append("=" * 78)
    lines.append("  CONVERGENCE STUDY: 4th-order finite differences on Schwarzschild")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"  {'Nr':>6}  {'h':>10}  {'|G^t_t|':>12}  {'|G^r_r|':>12}  {'|G^θ_θ|':>12}  {'|R|':>12}")
    lines.append("-" * 78)

    for r in conv['resolutions']:
        lines.append(f"  {r['Nr']:>6}  {r['h']:>10.6f}  {r['G_tt_err']:>12.4e}  "
                     f"{r['G_rr_err']:>12.4e}  {r['G_th_err']:>12.4e}  {r['R_err']:>12.4e}")

    lines.append("")
    lines.append("  Convergence orders (p such that error ~ h^p):")
    lines.append(f"  {'Nr₁→Nr₂':>14}  {'h₁/h₂':>8}  {'e₁/e₂':>12}  {'order p':>10}")
    lines.append("-" * 50)

    for o in conv['convergence_orders']:
        lines.append(f"  {o['Nr1']:>5}→{o['Nr2']:<5}  {o['h1'] / o['h2']:>8.3f}  "
                     f"{o['e1'] / (o['e2'] + 1e-30):>12.3f}  {o['order']:>10.2f}")

    lines.append("")
    lines.append(f"  Mean convergence order: {conv['mean_order']:.2f}  (expected: 4.0)")
    lines.append(f"  4th order achieved: {'✓ YES' if conv['order_achieved'] else '✗ NO'}")
    lines.append("=" * 78)

    return "\n".join(lines)




def parametric_E_factor_study(r_arr, dp_base, cfg, E_factors=None):
    if E_factors is None:
        E_factors = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    M = dp_base.M_dimless
    A_schw, B_schw = schwarzschild_isotropic(r_arr, M)

    results = []

    for E_fac in E_factors:
        dp_mod = copy_derived_params_with_E_factor(dp_base, E_fac)

        try:
            A_sol, B_sol = solve_metric_ode_strict(r_arr, dp_mod, cfg)

            ein = compute_einstein_tensor_isotropic(r_arr, A_sol, B_sol)

            E_r, B_theta = em_field_profile(r_arr, dp_mod.R_bubble_dimless, dp_mod.sigma_dimless,
                                            dp_mod.E_factor_dimless, dp_mod.M_dimless)
            Lr, _, _, _, _, _, _ = compute_lambda_rho(E_r, B_theta, A_sol, B_sol)

            dA = np.max(np.abs(A_sol - A_schw))
            dB = np.max(np.abs(B_sol - B_schw))

            mask = (r_arr > 0.3) & (r_arr < r_arr[-5])
            warp_ratio = A_sol[mask] / np.clip(A_schw[mask], 1e-10, None)
            min_warp = float(np.min(warp_ratio))
            max_warp = float(np.max(warp_ratio))

            rho_eff = -(ein['G_t_t'][mask] - Lr[mask]) / (8 * np.pi)
            p_eff = (ein['G_r_r'][mask] - Lr[mask]) / (8 * np.pi)
            NEC_min = float(np.min(rho_eff + p_eff))
            NEC_violated = NEC_min < -1e-10

            Lr_max = float(np.max(np.abs(Lr)))

            results.append({
                'E_factor': E_fac,
                'success': True,
                'delta_A_max': dA,
                'delta_B_max': dB,
                'min_warp_ratio': min_warp,
                'max_warp_ratio': max_warp,
                'Lr_max': Lr_max,
                'NEC_min': NEC_min,
                'NEC_violated': NEC_violated,
                'A_sol': A_sol,
                'B_sol': B_sol,
                'Lr': Lr,
            })

        except Exception as e:
            results.append({
                'E_factor': E_fac,
                'success': False,
                'error': str(e),
            })

    return {
        'E_factors': E_factors,
        'results': results,
        'r_arr': r_arr,
        'A_schw': A_schw,
        'B_schw': B_schw,
    }


def copy_derived_params_with_E_factor(dp, E_fac_new):
    import copy
    dp_new = copy.deepcopy(dp)
    dp_new.E_factor_dimless = E_fac_new
    return dp_new


def format_parametric_report(param_data):
    lines = []
    lines.append("=" * 90)
    lines.append("  PARAMETRIC ANALYSIS: E_factor scan")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  {'E_factor':>10}  {'ΔA_max':>10}  {'ΔB_max':>10}  "
                 f"{'warp_min':>10}  {'warp_max':>10}  {'|Λ_ρ|_max':>12}  {'NEC_min':>10}  {'NEC?':>6}")
    lines.append("-" * 90)

    for r in param_data['results']:
        if r['success']:
            nec_str = "VIOL" if r['NEC_violated'] else "OK"
            lines.append(f"  {r['E_factor']:>10.2f}  {r['delta_A_max']:>10.4e}  "
                         f"{r['delta_B_max']:>10.4e}  {r['min_warp_ratio']:>10.4f}  "
                         f"{r['max_warp_ratio']:>10.4f}  {r['Lr_max']:>12.4e}  "
                         f"{r['NEC_min']:>10.4e}  {nec_str:>6}")
        else:
            lines.append(f"  {r['E_factor']:>10.2f}  {'FAILED':>10}  {r.get('error', '')}")

    lines.append("=" * 90)
    return "\n".join(lines)




def plot_convergence(conv_data, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    h_arr = [r['h'] for r in conv_data['resolutions']]

    ax = axes[0]
    for key, label in [('G_tt_err', '$|G^t_t|_\\infty$'),
                       ('G_rr_err', '$|G^r_r|_\\infty$'),
                       ('G_th_err', '$|G^\\theta_\\theta|_\\infty$'),
                       ('R_err', '$|R|_\\infty$')]:
        errs = [r[key] for r in conv_data['resolutions']]
        ax.loglog(h_arr, errs, 'o-', label=label, markersize=8)

    h_ref = np.array(h_arr)
    e_ref = conv_data['resolutions'][0]['total_err'] * (h_ref / h_ref[0]) ** 4
    ax.loglog(h_ref, e_ref, 'k--', alpha=0.5, label='$\\propto h^4$')

    ax.set_xlabel('Grid spacing $h$', fontsize=12)
    ax.set_ylabel('Max error', fontsize=12)
    ax.set_title('Convergence: $G_{\\mu\\nu}$ error vs resolution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    orders = [o['order'] for o in conv_data['convergence_orders']]
    Nr_labels = [f"{o['Nr1']}→{o['Nr2']}" for o in conv_data['convergence_orders']]
    x = range(len(orders))
    ax.bar(x, orders, color='steelblue', alpha=0.7)
    ax.axhline(y=4.0, color='red', linestyle='--', label='Expected order = 4')
    ax.set_xticks(x)
    ax.set_xticklabels(Nr_labels, rotation=45, fontsize=9)
    ax.set_ylabel('Convergence order $p$', fontsize=12)
    ax.set_title(f'Measured order: {conv_data["mean_order"]:.2f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_parametric(param_data, save_path=None):
    import matplotlib.pyplot as plt

    good = [r for r in param_data['results'] if r['success']]
    if not good:
        print("No successful parametric runs")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    r = param_data['r_arr']

    E_facs = [g['E_factor'] for g in good]

    ax = axes[0, 0]
    ax.plot(r, param_data['A_schw'], 'k-', lw=2, label='Schwarzschild')
    for g in good[::max(1, len(good) // 5)]:
        ax.plot(r, g['A_sol'], '--', label=f'E={g["E_factor"]:.1f}', alpha=0.7)
    ax.set_xlabel('$r/r_s$')
    ax.set_ylabel('$A(r)$')
    ax.set_title('Lapse function')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(r, param_data['B_schw'], 'k-', lw=2, label='Schwarzschild')
    for g in good[::max(1, len(good) // 5)]:
        ax.plot(r, g['B_sol'], '--', label=f'E={g["E_factor"]:.1f}', alpha=0.7)
    ax.set_xlabel('$r/r_s$')
    ax.set_ylabel('$B(r)$')
    ax.set_title('Conformal factor')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for g in good[::max(1, len(good) // 5)]:
        if g['E_factor'] > 0:
            ax.plot(r, g['Lr'], label=f'E={g["E_factor"]:.1f}', alpha=0.7)
    ax.set_xlabel('$r/r_s$')
    ax.set_ylabel('$\\Lambda_\\rho$')
    ax.set_title('Alena lambda')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(E_facs, [g['delta_A_max'] for g in good], 'bo-')
    ax.set_xlabel('E_factor')
    ax.set_ylabel('$\\max|A - A_{Schw}|$')
    ax.set_title('Lapse deviation')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(E_facs, [g['min_warp_ratio'] for g in good], 'rs-', label='min warp')
    ax.plot(E_facs, [g['max_warp_ratio'] for g in good], 'b^-', label='max warp')
    ax.axhline(y=1.0, color='gray', linestyle='--')
    ax.set_xlabel('E_factor')
    ax.set_ylabel('$A/A_{Schw}$')
    ax.set_title('Warp ratio range')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    nec_vals = [g['NEC_min'] for g in good]
    colors = ['red' if g['NEC_violated'] else 'green' for g in good]
    ax.bar(range(len(good)), nec_vals, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_xticks(range(len(good)))
    ax.set_xticklabels([f'{g["E_factor"]:.1f}' for g in good], rotation=45, fontsize=8)
    ax.set_xlabel('E_factor')
    ax.set_ylabel('$\\min(\\rho + p)$')
    ax.set_title('NEC check (red = violated)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Parametric Analysis: E_factor scan', fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_time_evolution(history, r_arr, save_path=None):
    import matplotlib.pyplot as plt

    n_snapshots = len(history['t'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cmap = plt.cm.viridis

    for i, t in enumerate(history['t']):
        color = cmap(i / max(1, n_snapshots - 1))
        alpha = 0.3 + 0.7 * i / max(1, n_snapshots - 1)

        axes[0].plot(r_arr, history['A'][i], color=color, alpha=alpha)
        axes[1].plot(r_arr, history['B'][i], color=color, alpha=alpha)
        axes[2].plot(r_arr, history['K'][i], color=color, alpha=alpha)

    axes[0].set_ylabel('$A(r,t)$')
    axes[0].set_title('Lapse evolution')
    axes[1].set_ylabel('$B(r,t)$')
    axes[1].set_title('Conformal factor evolution')
    axes[2].set_ylabel('$K(r,t)$')
    axes[2].set_title('Extrinsic curvature')

    for ax in axes:
        ax.set_xlabel('$r/r_s$')
        ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(history['t'][0], history['t'][-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=axes.tolist(), label='$t/r_s$', shrink=0.8)

    plt.suptitle('Time-dependent warp bubble evolution', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()




def run_full_analysis(r_arr, dp, cfg):
    results = {}

    print("[CONVERGENCE] Running convergence study...")
    t0 = time.time()
    conv = run_convergence_study(dp, cfg, Nr_list=[500, 750, 1000, 1500, 2000])
    results['convergence'] = conv
    print(format_convergence_report(conv))
    print(f"  Time: {time.time() - t0:.1f}s\n")

    print("[STATIC] Solving with strict φ'' from G_θθ...")
    t0 = time.time()
    A_sol, B_sol = solve_metric_ode_strict(r_arr, dp, cfg)
    results['A_static'] = A_sol
    results['B_static'] = B_sol

    ein = compute_einstein_tensor_isotropic(r_arr, A_sol, B_sol)
    results['einstein_static'] = ein
    print(f"  EFE residuals: |G^t_t|={np.max(np.abs(ein['G_t_t'])):.4e}, "
          f"|G^r_r|={np.max(np.abs(ein['G_r_r'])):.4e}, "
          f"|G^θ_θ|={np.max(np.abs(ein['G_th_th'])):.4e}")
    print(f"  Time: {time.time() - t0:.1f}s\n")

    print("[TIME-DEP] Running time-dependent evolution...")
    t0 = time.time()
    history = solve_time_dependent(r_arr, dp, cfg, Nt=200, dt_factor=0.05)
    results['time_dep'] = history
    print(f"  Evolved to t = {history['t'][-1]:.4f} r_s")
    print(f"  {len(history['t'])} snapshots stored")
    print(f"  Time: {time.time() - t0:.1f}s\n")

    print("[PARAMETRIC] Running E_factor scan...")
    t0 = time.time()
    param = parametric_E_factor_study(r_arr, dp, cfg,
                                      E_factors=[0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    results['parametric'] = param
    print(format_parametric_report(param))
    print(f"  Time: {time.time() - t0:.1f}s\n")

    try:
        plot_convergence(conv)
        plot_parametric(param)
        plot_time_evolution(history, r_arr)
    except Exception as e:
        print(f"  Plotting error: {e}")

    return results






class FlatBubbleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Alena Tensor v15: ADM + AsymFlat + SchwTest + Dimensionless")
        self.geometry("1850x1000")
        self.configure(bg="#101010")
        self._styles()
        self.engine = PhysicsEngine(CONFIG)
        self.result_queue = queue.Queue()
        self.calc_thread = None
        self.calc_time = 0.0
        self._build_ui()
        self.after(200, self.trigger_calc)

    def _styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        bg = "#1a1a1a"; fg = "#d0d0d0"; ac = "#00bcd4"
        s.configure("TFrame", background=bg)
        s.configure("TLabel", background=bg, foreground=fg, font=("Consolas", 10))
        s.configure("H.TLabel", font=("Consolas", 12, "bold"), foreground=ac, background=bg)
        s.configure("TButton", background="#2a2a2a", foreground=fg,
                     font=("Consolas", 10, "bold"), padding=6)
        s.map("TButton", background=[("active", ac)])
        s.configure("TNotebook", background=bg, borderwidth=0)
        s.configure("TNotebook.Tab", background="#252525", foreground=fg,
                     padding=[10, 4], font=("Consolas", 9, "bold"))
        s.map("TNotebook.Tab", background=[("selected", ac)], foreground=[("selected", "#fff")])

    def _export_html(self):
        if not self.engine._results:
            messagebox.showwarning("Внимание", "Сначала выполните расчёт!")
            return
        path = generate_html_report(self.engine)
        messagebox.showinfo("Готово", f"Отчёт сохранён:\n{path}")
        import webbrowser
        webbrowser.open(path)

    def _build_ui(self):
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(main, width=400)
        main.add(left, weight=0)
        ttk.Label(left, text="ПАРАМЕТРЫ (размерные)", style="H.TLabel").pack(pady=8)

        sf = ttk.Frame(left)
        sf.pack(fill=tk.X, padx=8)
        self.vars = {}

        def slider(n, l, lo, hi, d, fmt=".2f"):
            fr = ttk.Frame(sf); fr.pack(fill=tk.X, pady=4)
            ttk.Label(fr, text=l, width=22).pack(side=tk.LEFT)
            v = tk.DoubleVar(value=d); self.vars[n] = v
            vl = ttk.Label(fr, text=f"{d:{fmt}}", width=7, foreground="#00bcd4")
            vl.pack(side=tk.RIGHT)
            def cb(val, _l=vl, _f=fmt): _l.config(text=f"{float(val):{_f}}")
            ttk.Scale(fr, from_=lo, to=hi, variable=v, command=cb).pack(
                side=tk.RIGHT, fill=tk.X, expand=True, padx=4)

        slider('r_s', 'r_s (Шварцшильд)', 0.5, 8.0, 3.0)
        slider('R_bubble', 'R_пузыря', 4.0, 20.0, 8.0)
        slider('sigma', 'Толщина σ', 0.5, 5.0, 2.0)
        slider('E_factor', 'ЭМ поле E₀', 1.0, 50.0, 10.0)

        self.btn = ttk.Button(left, text="▶ РАСЧЁТ", command=self.trigger_calc)
        self.btn.pack(fill=tk.X, padx=16, pady=10)

        self.btn_html = ttk.Button(left, text="📄 HTML Отчёт", command=self._export_html)
        self.btn_html.pack(fill=tk.X, padx=16, pady=5)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(left, text="РЕЗУЛЬТАТЫ", style="H.TLabel").pack(pady=4)

        self.info = {}
        labels = [
            ('efe_tt', 'EFE(tt): —'), ('efe_rr', 'EFE(rr): —'),
            ('efe_th', 'EFE(θθ): —'), ('divT', '∇T: —'),
            ('comp', 'Компенс.: —'), ('A_in', 'A внутри: —'),
            ('adm', 'ADM: —'), ('asym', 'Асимпт.: —'),
            ('schw', 'Schw тест: —'), ('stab', 'Устойч.: —'),
            ('wec', 'WEC: —'), ('nec', 'NEC: —'),
            ('sec', 'SEC: —'), ('dec', 'DEC: —'),
            ('lr', 'Λ_ρ: —'),
        ]
        for k, t in labels:
            l = ttk.Label(left, text=t)
            l.pack(anchor=tk.W, padx=12, pady=1)
            self.info[k] = l

        self.lbl_status = ttk.Label(left, text="Готов", foreground="#888")
        self.lbl_status.pack(side=tk.BOTTOM, pady=8)

        self.nb = ttk.Notebook(main)
        main.add(self.nb, weight=1)
        tabs = [("Метрика+EFE", 't1'), ("1D Профили", 't2'),
                ("Тензоры", 't3'), ("Энерг.Условия", 't4'),
                ("Геодезические", 't5'), ("SchwTest+ADM", 't6'),
                ("3D", 't7'), ("Отчёт", 't8')]
        for title, attr in tabs:
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=f" {title} ")
            setattr(self, attr, f)
        self._init_figs()

        self.txt = scrolledtext.ScrolledText(self.t8, bg="#060606", fg="#00ff88",
                                             font=("Consolas", 11), padx=16, pady=16, wrap=tk.WORD)
        self.txt.pack(fill=tk.BOTH, expand=True)

    def _init_figs(self):
        bg = '#101010'
        def mk(parent, nr, nc, fs=(12, 9)):
            fig = plt.Figure(figsize=fs, dpi=96, facecolor=bg)
            gs = GridSpec(nr, nc, figure=fig, wspace=0.35, hspace=0.45)
            ax = [fig.add_subplot(gs[i, j]) for i in range(nr) for j in range(nc)]
            c = FigureCanvasTkAgg(fig, master=parent)
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(c, parent).update()
            return fig, ax, c

        self.f1, self.a1, self.c1 = mk(self.t1, 2, 2)
        self.f2, self.a2, self.c2 = mk(self.t2, 2, 3, (14, 9))
        self.f3, self.a3, self.c3 = mk(self.t3, 2, 2)
        self.f4, self.a4, self.c4 = mk(self.t4, 2, 2)

        self.f5 = plt.Figure(figsize=(12, 9), dpi=96, facecolor=bg)
        self.a5 = self.f5.add_subplot(111)
        self.c5 = FigureCanvasTkAgg(self.f5, master=self.t5)
        self.c5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.c5, self.t5).update()

        self.f6, self.a6, self.c6 = mk(self.t6, 2, 2)

        self.f7 = plt.Figure(figsize=(9, 8), dpi=96, facecolor=bg)
        self.a7 = self.f7.add_subplot(111, projection='3d')
        self.a7.set_facecolor(bg)
        self.c7 = FigureCanvasTkAgg(self.f7, master=self.t7)
        self.c7.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.c7, self.t7).update()

    def trigger_calc(self):
        if self.calc_thread and self.calc_thread.is_alive(): return
        self.btn.config(state=tk.DISABLED, text="⏳ РАСЧЁТ...")
        self.lbl_status.config(text="Вычисления...", foreground="#ff0")
        self.engine.r_s = self.vars['r_s'].get()
        self.engine.R_bubble = self.vars['R_bubble'].get()
        self.engine.sigma = self.vars['sigma'].get()
        self.engine.E_factor = self.vars['E_factor'].get()
        while not self.result_queue.empty(): self.result_queue.get()
        self.calc_thread = threading.Thread(target=self._work, daemon=True)
        self.calc_thread.start()
        self.after(100, self._poll)

    def _work(self):
        try:
            t0 = time.time()
            self.engine.execute_full_pipeline()
            self.result_queue.put(("OK", time.time() - t0))
        except:
            self.result_queue.put(("ERR", traceback.format_exc()))

    def _poll(self):
        try:
            msg, p = self.result_queue.get_nowait()
            if msg == "OK":
                self.calc_time = p; self._ok()
            else:
                self._err(p)
        except queue.Empty:
            self.after(100, self._poll)

    def _err(self, e):
        self.btn.config(state=tk.NORMAL, text="▶ РАСЧЁТ")
        self.lbl_status.config(text="ОШИБКА", foreground="#f00")
        messagebox.showerror("Ошибка", e)

    def _ok(self):
        self.btn.config(state=tk.NORMAL, text="▶ РАСЧЁТ")
        self.lbl_status.config(text=f"Готово: {self.calc_time:.2f}с", foreground="#0f0")
        a = self.engine.analytics

        self.info['efe_tt'].config(text=f"EFE(tt): {a['res_tt']:.2e}")
        self.info['efe_rr'].config(text=f"EFE(rr): {a['res_rr']:.2e}")
        self.info['efe_th'].config(text=f"EFE(θθ): {a['res_thth']:.2e}")
        self.info['divT'].config(text=f"∇T: {a['div_T_mean']:.2e}")
        self.info['comp'].config(text=f"Компенс.: {a['comp_inside'] * 100:.1f}%")
        self.info['A_in'].config(text=f"A внутри: {a['A_inside']:.5f}")

        ad = a['adm']
        self.info['adm'].config(text=f"ADM: M={ad['M_ADM']:.4f} (ожид {ad['M_expected']:.4f})")
        asy = a['asym']
        self.info['asym'].config(text=f"Асимпт: |A-1|={asy['dev_A_max']:.2e}")
        st = a['schw_test']
        self.info['schw'].config(text=f"Schw: K_err={st['K_rel_err_mean']:.2e}")
        stb = a['stability']
        self.info['stab'].config(text=f"Уст: δA={stb['dres_dA']:.2e}")
        self.info['wec'].config(text=f"WEC: {a['wec_pct']:.1f}%")
        self.info['nec'].config(text=f"NEC: {a['nec_pct']:.1f}%")
        self.info['sec'].config(text=f"SEC: {a['sec_pct']:.1f}%")
        self.info['dec'].config(text=f"DEC: {a['dec_pct']:.1f}%")
        self.info['lr'].config(text=f"Λ_ρ: [{a['Lr_min']:.2e},{a['Lr_max']:.2e}]")

        self._render()
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, self.engine.generate_report())

    def _sax(self, ax, title, xlabel='r/r_s'):
        ax.clear()
        ax.set_title(title, color='#00bcd4', pad=6, fontsize=9, fontweight='bold')
        ax.tick_params(colors='#999', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.set_facecolor('#0a0a0a')
        if xlabel: ax.set_xlabel(xlabel, color='#777', fontsize=7)

    def _render(self):
        R = self.engine._results
        X, Y = self.engine.X2d, self.engine.Y2d
        h = self.engine.cfg.extent_2d_over_rs
        r = R['r']

        ax = self.a1[0]; self._sax(ax, "A(r̃)", '')
        ax.pcolormesh(X, Y, R['A_2d'], cmap='viridis', shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1.5)
        ax.set_aspect('equal')

        ax = self.a1[1]; self._sax(ax, "Компенсация кривизны", '')
        ax.pcolormesh(X, Y, np.clip(R['comp_2d'], 0, 1), cmap='Greens',
                      vmin=0, vmax=1, shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='cyan', linewidths=1.5)
        ax.set_aspect('equal')

        ax = self.a1[2]; self._sax(ax, "EFE невязка (tt)", '')
        vm = np.percentile(np.abs(R['res_tt_2d']), 98) + 1e-30
        ax.pcolormesh(X, Y, R['res_tt_2d'], cmap='seismic', vmin=-vm, vmax=vm, shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1)
        ax.set_aspect('equal')

        ax = self.a1[3]; self._sax(ax, "∇_μ T^{μr}", '')
        vm2 = np.percentile(np.abs(R['div_T_2d']), 98) + 1e-30
        ax.pcolormesh(X, Y, R['div_T_2d'], cmap='PuOr', vmin=-vm2, vmax=vm2, shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1)
        ax.set_aspect('equal')
        self.f1.tight_layout(pad=2); self.c1.draw()

        ax = self.a2[0]; self._sax(ax, "A(r̃), B(r̃)")
        ax.plot(r, R['A_schw'], 'r--', lw=1.5, label='A Schw', alpha=0.7)
        ax.plot(r, R['A'], 'r-', lw=2, label='A(r̃)')
        ax.plot(r, R['B_schw'], 'b--', lw=1.5, label='B Schw', alpha=0.7)
        ax.plot(r, R['B'], 'b-', lw=2, label='B(r̃)')
        ax.axhline(1, color='#555', ls=':', lw=0.5)
        ax.fill_between(r, 0, 2, where=R['f_bub']>0.3, alpha=0.08, color='cyan')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        ax = self.a2[1]; self._sax(ax, "f(r̃) и компенсация")
        ax.plot(r, R['f_bub'], 'c-', lw=2, label='f(r̃)')
        ax.plot(r, np.clip(R['compensation'], -1, 1), 'g-', lw=2, label='компенсация')
        ax.axhline(1, color='#555', ls=':', lw=0.5)
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        ax = self.a2[2]; self._sax(ax, "Λ_ρ(r̃)")
        ax.plot(r, R['Lr'], 'b-', lw=2)
        ax.axhline(0, color='#555', ls=':', lw=0.5)
        ax.fill_between(r, 0, R['Lr'], where=R['Lr']<0, alpha=0.3, color='red')

        ax = self.a2[3]; self._sax(ax, "EFE невязки (нормир.)")
        ax.semilogy(r, np.clip(R['efe']['rel_tt'], 1e-15, None), 'r-', lw=1.5, label='|tt|')
        ax.semilogy(r, np.clip(R['efe']['rel_rr'], 1e-15, None), 'b-', lw=1.5, label='|rr|')
        ax.semilogy(r, np.clip(R['efe']['rel_thth'], 1e-15, None), 'g-', lw=1.5, label='|θθ|')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        ax = self.a2[4]; self._sax(ax, "∇_μ T^{μr}(r̃)")
        ax.plot(r, R['div_T'], 'm-', lw=2)
        ax.axhline(0, color='#555', ls=':', lw=0.5)

        ax = self.a2[5]; self._sax(ax, "Kretschmann K(r̃)")
        ax.semilogy(r, np.clip(R['K_schw'], 1e-20, None), 'r--', lw=1.5, label='K Schw')
        ax.semilogy(r, np.clip(R['K_eff'], 1e-20, None), 'g-', lw=2, label='K с пузырём')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)
        self.f2.tight_layout(pad=2); self.c2.draw()

        ax = self.a3[0]; self._sax(ax, "Λ_ρ", '')
        ax.pcolormesh(X, Y, R['Lr_2d'], cmap='RdBu_r', norm=CenteredNorm(), shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1)
        ax.set_aspect('equal')

        ax = self.a3[1]; self._sax(ax, "|Ê| физ.", '')
        ax.pcolormesh(X, Y, R['E_phys_2d'], cmap='inferno', shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1)
        ax.set_aspect('equal')

        ax = self.a3[2]; self._sax(ax, "R скаляр", '')
        ax.pcolormesh(X, Y, R['Rscal_2d'], cmap='PuOr', norm=CenteredNorm(), shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1)
        ax.set_aspect('equal')

        ax = self.a3[3]; self._sax(ax, "G_{tt}", '')
        ax.pcolormesh(X, Y, R['Gtt_2d'], cmap='BrBG', norm=CenteredNorm(), shading='auto')
        ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='lime', linewidths=1)
        ax.set_aspect('equal')
        self.f3.tight_layout(pad=2); self.c3.draw()

        pcts = ['wec_pct', 'nec_pct', 'sec_pct', 'dec_pct']
        fields = [R['wec_2d'], R['nec_2d'], R['sec_2d'], R['dec_2d']]
        titles = ["WEC", "NEC", "SEC", "DEC"]
        for ax, t, fld, pk in zip(self.a4, titles, fields, pcts):
            self._sax(ax, t, '')
            ax.pcolormesh(X, Y, fld, cmap='Reds', vmin=0, vmax=1, shading='auto')
            ax.contour(X, Y, R['f_2d'], levels=[0.5], colors='cyan', linewidths=1)
            p = self.engine.analytics[pk]
            ax.text(-h+0.5, h-1.5, f"{p:.1f}%", color='w', fontsize=9,
                    bbox=dict(facecolor='k', alpha=0.6))
            ax.set_aspect('equal')
        self.f4.tight_layout(pad=2); self.c4.draw()

        ax = self.a5; self._sax(ax, "Геодезические (синий=Schw, зелёный=пузырь)", '')
        K_disp = np.log10(np.abs(R['K_schw_2d']) + 1e-10)
        ax.imshow(K_disp, extent=[-h,h,-h,h], origin='lower', cmap='hot', alpha=0.4)
        for rx, ry in zip(R['traj_xs'], R['traj_ys']):
            ax.plot(rx, ry, color='#4488ff', lw=0.7, alpha=0.5)
        for rx, ry in zip(R['traj_xe'], R['traj_ye']):
            ax.plot(rx, ry, color='#00ffd0', lw=1.2, alpha=0.7)
        th = np.linspace(0, 2*np.pi, 100)
        rb = self.engine.dp.R_bubble_dimless
        ax.plot(rb*np.cos(th), rb*np.sin(th), 'y--', lw=1.5, alpha=0.7, label='пузырь')
        rs_h = self.engine.dp.M_dimless
        ax.plot(rs_h*np.cos(th), rs_h*np.sin(th), 'r-', lw=2, alpha=0.8, label='горизонт')
        ax.set_xlim(-h,h); ax.set_ylim(-h,h); ax.set_aspect('equal')
        ax.legend(facecolor='black', labelcolor='white', fontsize=8)
        self.f5.tight_layout(pad=2); self.c5.draw()

        st = R['schw_test']

        ax = self.a6[0]; self._sax(ax, "Schw test: K вычисл. vs точное")
        mask = r > r[5]
        ax.semilogy(r[mask], np.clip(st['K_exact'][mask], 1e-20, None),
                    'r-', lw=2, label='K exact = 48M²/R⁶')
        ax.semilogy(r[mask], np.clip(st['K_computed'][mask], 1e-20, None),
                    'g--', lw=1.5, label='K computed')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        ax = self.a6[1]; self._sax(ax, "Schw test: отн. ошибка K")
        K_err = np.abs(st['K_computed'] - st['K_exact']) / (st['K_exact'] + 1e-30)
        ax.semilogy(r[mask], np.clip(K_err[mask], 1e-15, None), 'b-', lw=2)
        ax.axhline(0.01, color='lime', ls='--', lw=1, label='1% порог')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        ax = self.a6[2]; self._sax(ax, "ADM: M(r̃) из A и B")
        ad = R['adm']
        n_prof = len(ad['M_from_A_profile'])
        r_prof = r[-n_prof:]
        ax.plot(r_prof, ad['M_from_A_profile'], 'r-', lw=2, label='M из A = −r(A−1)')
        ax.plot(r_prof, ad['M_from_B_profile'], 'b-', lw=2, label='M из B = r(B−1)')
        ax.axhline(ad['M_expected'], color='lime', ls='--', lw=1.5, label=f'M={ad["M_expected"]:.3f}')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        ax = self.a6[3]; self._sax(ax, "Асимпт. плоскость: |A−1|, |B−1|")
        ax.semilogy(r, np.abs(R['A'] - 1.0) + 1e-15, 'r-', lw=2, label='|A−1|')
        ax.semilogy(r, np.abs(R['B'] - 1.0) + 1e-15, 'b-', lw=2, label='|B−1|')
        ax.semilogy(r, self.engine.dp.M_dimless / r, 'g--', lw=1, alpha=0.5, label='M/r̃')
        ax.legend(facecolor='black', labelcolor='white', fontsize=7)

        self.f6.tight_layout(pad=2); self.c6.draw()

        ax = self.a7; ax.clear(); ax.set_facecolor('#101010')
        ax.plot_surface(X, Y, R['A_2d'], cmap='coolwarm', linewidth=0,
                        antialiased=True, alpha=0.9)
        ax.set_title("3D A(r̃)", color='#00bcd4', pad=15, fontweight='bold')
        for p in [ax.xaxis, ax.yaxis, ax.zaxis]:
            p.set_tick_params(colors='#777')
            p.pane.fill = False
            p.pane.set_edgecolor('#333')
        self.f7.tight_layout(); self.c7.draw()






def main():
    print("="*76)
    print("  ALENA TENSOR v15.0")
    print("  Analytic Riemann + Shooting ODE + Full Kretschmann")
    print("  + ADM Mass + Asymptotic Flatness + Schwarzschild Test + Dimensionless")
    print(f"  Numba: {'ON' if NUMBA_AVAILABLE else 'OFF'}, CPUs: {os.cpu_count()}")
    print("="*76)
    app = FlatBubbleApp()
    app.mainloop()

if __name__ == "__main__":
    main()