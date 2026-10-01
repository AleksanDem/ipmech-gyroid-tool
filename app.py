import os
import tempfile
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
import math
import trimesh
import trimesh.repair
from skimage import measure
from scipy import ndimage


# 1. Настройка страницы
st.set_page_config(
    page_title="IPMech TPMS Lattice Tool",
    layout="wide"
)

# --- СИСТЕМА ЛОКАЛИЗАЦИИ (RU / EN) ---
TRANSLATIONS = {
    "cell_params":    {"ru": "⚙️ Параметры ячейки",  "en": "⚙️ Cell Parameters"},
    "model_params":   {"ru": "📦 Параметры модели",   "en": "📦 Model Parameters"},
    "lattice_type_label":{"ru": "Тип структуры",       "en": "Lattice Type"},
    "lattice_type_help": {"ru": "Геометрический тип TPMS поверхности (минимальной поверхности)",
                          "en": "Geometric type of TPMS surface (triply periodic minimal surface)"},
    "cell_size_label":{"ru": "Период ячейки, мм",     "en": "Cell period, mm"},
    "cell_size_help": {"ru": "Длина одного периода ячейки (мм). Меньше = более частая решётка",
                       "en": "One cell period length (mm). Smaller = finer lattice"},
    "wall_label":     {"ru": "Толщина стенки, мм",    "en": "Wall thickness, mm"},
    "wall_help":      {"ru": "Целевая толщина стенок структуры (мм, калибруется через градиент)",
                       "en": "Target wall thickness of structure (mm, calibrated via gradient)"},
    "size_x_label":   {"ru": "Размер X, мм",           "en": "Size X, mm"},
    "size_y_label":   {"ru": "Размер Y, мм",           "en": "Size Y, mm"},
    "size_z_label":   {"ru": "Размер Z, мм",           "en": "Size Z, mm"},
    "size_x_help":    {"ru": "Размер блока по оси X (мм)", "en": "Block size along X axis (mm)"},
    "size_y_help":    {"ru": "Размер блока по оси Y (мм)", "en": "Block size along Y axis (mm)"},
    "size_z_help":    {"ru": "Размер блока по оси Z (мм)", "en": "Block size along Z axis (mm)"},
    "res_label":      {"ru": "Разрешение, вок/яч",     "en": "Resolution, vox/cell"},
    "res_help":       {"ru": "Число вокселей на одну гироидную ячейку. Больше = точнее, но медленнее",
                       "en": "Number of voxels per unit cell. Higher = finer mesh, slower"},
    "clean_islands_label":{"ru": "Удалять несвязные углы", "en": "Filter stray fragments"},
    "clean_islands_help": {"ru": "Исключает изолированные мелкие осколки и несвязные уголки на границах блока",
                           "en": "Excludes isolated fragments and stray corners at the block boundaries"},
    "ro_label":       {"ru": "Плотность Ro, г/см³",    "en": "Density Ro, g/cm³"},
    "ro_help":        {"ru": "Физическая плотность материала (г/см³)",
                       "en": "Material density (g/cm³)"},
    "btn_generate":   {"ru": "🛠️ Подготовить STL",    "en": "🛠️ Generate STL"},
    "btn_download":   {"ru": "📥 Скачать STL",         "en": "📥 Download STL"},
    "generating":     {"ru": "Генерация STL...",        "en": "Generating STL..."},
    "structure":      {"ru": "📈 Структура",            "en": "📈 Structure"},
    "tab_3d":         {"ru": "3D Просмотр",             "en": "3D Preview"},
    "tab_slice":      {"ru": "Срез XY",                 "en": "XY Slice"},
    "3d_stale":       {"ru": "⚠️ Параметры были изменены. Нажмите '🛠️ Подготовить STL' для обновления 3D модели.",
                       "en": "⚠️ Parameters changed. Click '🛠️ Generate STL' to update the 3D model."},
    "3d_empty":       {"ru": "Сгенерируйте STL (кнопка слева), чтобы увидеть 3D превью.",
                       "en": "Generate STL (button on the left) to see 3D preview."},
    "metrics":        {"ru": "📊 Характеристики",      "en": "📊 Metrics"},
    "m_lattice":      {"ru": "Решетка",                 "en": "Lattice"},
    "m_size_x":       {"ru": "Размер X",                "en": "Size X"},
    "m_size_y":       {"ru": "Размер Y",                "en": "Size Y"},
    "m_size_z":       {"ru": "Размер Z",                "en": "Size Z"},
    "m_cell":         {"ru": "Период ячейки",           "en": "Cell period"},
    "m_wall":         {"ru": "Толщина стенки",          "en": "Wall thickness"},
    "m_cells_x":      {"ru": "Ячеек X",                 "en": "Cells X"},
    "m_cells_y":      {"ru": "Ячеек Y",                 "en": "Cells Y"},
    "m_cells_z":      {"ru": "Ячеек Z",                 "en": "Cells Z"},
    "m_volume_full":  {"ru": "V полный",                "en": "V full"},
    "m_fill":         {"ru": "Заполнение",              "en": "Fill Ratio"},
    "m_mass":         {"ru": "Масса",                   "en": "Mass"},
    "m_threshold":    {"ru": "Порог t",                 "en": "Threshold t"},
    "m_ro_mat":       {"ru": "Плотность Ro",            "en": "Density Ro"},
    "mm":             {"ru": "мм",                      "en": "mm"},
    "mm3":            {"ru": "мм³",                     "en": "mm³"},
    "g":              {"ru": "г",                       "en": "g"},
    "gcm3":           {"ru": "г/см³",                   "en": "g/cm³"},
}

if 'lang' not in st.session_state:
    st.session_state['lang'] = 'ru'

if st.session_state.get('lang_toggle', False):
    st.session_state['lang'] = 'en'
else:
    st.session_state['lang'] = 'ru'

def t(key):
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(st.session_state['lang'], entry.get('ru', key))

# --- CSS СТИЛИЗАЦИЯ ИНТЕРФЕЙСА ---
st.markdown("""
    <style>
           .block-container {
               padding-top: 0.5rem !important;
               padding-bottom: 0.5rem !important;
               padding-left: 1.5rem !important;
               padding-right: 1.5rem !important;
           }
           header { visibility: hidden; }
           div[data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
           div[data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
           div[data-testid="stElementContainer"] { margin-bottom: 0px !important; }
           .stTabs [data-baseweb="tab-list"] { gap: 8px !important; }
           .stTabs [data-baseweb="tab"] {
               padding-top: 4px !important;
               padding-bottom: 4px !important;
               font-size: 0.85rem !important;
           }
           div[data-testid="stNumberInputContainer"],
           [data-testid="stNumberInput"] > div {
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
               display: flex !important;
               align-items: center !important;
           }
           div[data-baseweb="base-input"],
           div[data-baseweb="input"],
           div[data-testid="stNumberInputContainer"] div[data-baseweb="input"] {
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
               align-items: center !important;
           }
           div[data-testid="stNumberInputContainer"] input,
           .stNumberInput input {
               margin: 0 !important;
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
               padding-top: 0px !important;
               padding-bottom: 0px !important;
               font-size: 0.85rem !important;
               align-self: center !important;
           }
           div[data-testid="stSelectbox"] {
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
               display: flex !important;
               align-items: center !important;
           }
           div[data-baseweb="select"] {
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
           }
           div[data-baseweb="select"] > div {
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
               padding-top: 0px !important;
               padding-bottom: 0px !important;
               font-size: 0.85rem !important;
           }
           .stNumberInput button {
               height: 28px !important;
               min-height: 28px !important;
               max-height: 28px !important;
               width: 28px !important;
               padding: 0 !important;
               display: flex !important;
               align-items: center !important;
               justify-content: center !important;
           }
          .section-header {
              margin-top: 8px !important;
              margin-bottom: 0px !important;
              font-size: 0.9rem !important;
              font-weight: bold;
              color: #5c88be;
              border-bottom: 1px solid #464b5d;
              padding-bottom: 5px;
          }
        .label-col { font-size: 0.85rem; color: #9ea4b0; padding-top: 4px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 10px; }
        .metric-box {
            background-color: #1e2129;
            border: 1px solid #3d4455;
            padding: 4px 2px;
            border-radius: 4px;
            text-align: center;
            min-height: 52px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .m-label { color: #9ea4b0; font-size: 0.62rem; text-transform: uppercase; line-height: 1.1; margin-bottom: 2px; }
        .m-value { color: #ffffff; font-size: 0.85rem; font-weight: bold; font-family: 'Consolas', monospace; }
        .m-unit { font-size: 0.6rem; color: #5c88be; margin-left: 1px; }
        div[data-testid="stCheckbox"], div[data-testid="stToggle"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-top: 0px !important;
            height: 28px !important;
        }
        .lang-label-ru { font-size: 0.85rem; font-weight: bold; height: 28px; display: flex; align-items: center; justify-content: flex-end; }
        .lang-label-en { font-size: 0.85rem; font-weight: bold; height: 28px; display: flex; align-items: center; justify-content: flex-start; }
        .column-footer { text-align: center; color: #808495; padding-top: 10px; font-size: 0.75rem; border-top: 1px solid #464b5d; margin-top: 10px; }
        .stButton > button { margin-bottom: -10px; }
    </style>
    """, unsafe_allow_html=True)


# --- МАТЕМАТИЧЕСКИЕ МОДЕЛИ TPMS ПОВЕРХНОСТЕЙ ---

def gyroid_field(X, Y, Z):
    return (
        np.sin(X) * np.cos(Y)
        + np.sin(Y) * np.cos(Z)
        + np.sin(Z) * np.cos(X)
    )

def gyroid_grad(X, Y, Z):
    dFx = np.cos(X)*np.cos(Y) - np.sin(Z)*np.sin(X)
    dFy = -np.sin(X)*np.sin(Y) + np.cos(Y)*np.cos(Z)
    dFz = -np.sin(Y)*np.sin(Z) + np.cos(Z)*np.cos(X)
    return np.sqrt(dFx*dFx + dFy*dFy + dFz*dFz)

def diamond_field(X, Y, Z):
    return (
        np.sin(X) * np.sin(Y) * np.sin(Z)
        + np.sin(X) * np.cos(Y) * np.cos(Z)
        + np.cos(X) * np.sin(Y) * np.cos(Z)
        + np.cos(X) * np.cos(Y) * np.sin(Z)
    )

def diamond_grad(X, Y, Z):
    dFx = (np.cos(X)*np.sin(Y)*np.sin(Z) + np.cos(X)*np.cos(Y)*np.cos(Z) -
           np.sin(X)*np.sin(Y)*np.cos(Z) - np.sin(X)*np.cos(Y)*np.sin(Z))
    dFy = (np.sin(X)*np.cos(Y)*np.sin(Z) - np.sin(X)*np.sin(Y)*np.cos(Z) +
           np.cos(X)*np.cos(Y)*np.cos(Z) - np.cos(X)*np.sin(Y)*np.sin(Z))
    dFz = (np.sin(X)*np.sin(Y)*np.cos(Z) - np.sin(X)*np.cos(Y)*np.sin(Z) -
           np.cos(X)*np.sin(Y)*np.sin(Z) + np.cos(X)*np.cos(Y)*np.cos(Z))
    return np.sqrt(dFx*dFx + dFy*dFy + dFz*dFz)

def primitive_field(X, Y, Z):
    return np.cos(X) + np.cos(Y) + np.cos(Z)

def primitive_grad(X, Y, Z):
    return np.sqrt(np.sin(X)**2 + np.sin(Y)**2 + np.sin(Z)**2)

def neovius_field(X, Y, Z):
    return 3 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4 * np.cos(X) * np.cos(Y) * np.cos(Z)

def neovius_grad(X, Y, Z):
    dFx = -3*np.sin(X) - 4*np.sin(X)*np.cos(Y)*np.cos(Z)
    dFy = -3*np.sin(Y) - 4*np.cos(X)*np.sin(Y)*np.cos(Z)
    dFz = -3*np.sin(Z) - 4*np.cos(X)*np.cos(Y)*np.sin(Z)
    return np.sqrt(dFx*dFx + dFy*dFy + dFz*dFz)

def iwp_field(X, Y, Z):
    return (2 * (np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z) + np.cos(Z)*np.cos(X))
            - (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z)))

def iwp_grad(X, Y, Z):
    dFx = -2*np.sin(X)*(np.cos(Y) + np.cos(Z)) + 2*np.sin(2*X)
    dFy = -2*np.sin(Y)*(np.cos(X) + np.cos(Z)) + 2*np.sin(2*Y)
    dFz = -2*np.sin(Z)*(np.cos(X) + np.cos(Y)) + 2*np.sin(2*Z)
    return np.sqrt(dFx*dFx + dFy*dFy + dFz*dFz)


LATTICE_SURFACES = {
    "gyroid": {
        "func": gyroid_field,
        "grad": gyroid_grad,
        "name_ru": "Gyroid (Schoen G)",
        "name_en": "Gyroid (Schoen G)",
        "tag": "G",
        "desc_ru": "Изотропная структура со спиральными непрерывными каналами",
        "desc_en": "Isotropic structure with continuous helical channels",
    },
    "diamond": {
        "func": diamond_field,
        "grad": diamond_grad,
        "name_ru": "Diamond (Schwarz D)",
        "name_en": "Diamond (Schwarz D)",
        "tag": "D",
        "desc_ru": "Высочайшая прочность и модуль упругости на осевое сжатие",
        "desc_en": "Maximum compressive stiffness and strength",
    },
    "primitive": {
        "func": primitive_field,
        "grad": primitive_grad,
        "name_ru": "Primitive (Schwarz P)",
        "name_en": "Primitive (Schwarz P)",
        "tag": "P",
        "desc_ru": "Кубическая симметрия с прямолинейными открытыми порами",
        "desc_en": "Cubic symmetry with straight orthogonal open pores",
    },
    "neovius": {
        "func": neovius_field,
        "grad": neovius_grad,
        "name_ru": "Neovius",
        "name_en": "Neovius",
        "tag": "Neo",
        "desc_ru": "Высокая площадь удельной поверхности для теплообмена и фильтрации",
        "desc_en": "High specific surface area for heat exchange and filtration",
    },
    "iwp": {
        "func": iwp_field,
        "grad": iwp_grad,
        "name_ru": "I-WP (Schoen)",
        "name_en": "I-WP (Schoen)",
        "tag": "IWP",
        "desc_ru": "Оптимальна для высоких плотностей и ударостойких сэндвич-панелей",
        "desc_en": "Optimal for high densities and impact-resistant panels",
    },
}

LATTICE_KEYS = list(LATTICE_SURFACES.keys())


@st.cache_data
def compute_t_and_mean_grad(lattice_type, cell_size, wall_thickness):
    """Калибрует пороговое значение t на основе среднего градиента поверхности."""
    n = 40
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    surf = LATTICE_SURFACES.get(lattice_type, LATTICE_SURFACES["gyroid"])
    F = surf["func"](X, Y, Z)
    G = surf["grad"](X, Y, Z)
    near_surface = np.abs(F) < 0.15
    if np.any(near_surface):
        mean_grad = float(np.mean(G[near_surface]))
    else:
        mean_grad = float(np.mean(G))
    mean_grad = max(mean_grad, 1e-4)
    t = wall_thickness * np.pi * mean_grad / cell_size
    return t, mean_grad


@st.cache_data
def compute_slice(lattice_type, size_x, size_y, cell_size_mm, wall_mm, res_xy=300):
    """Вычисляет 2D-срез неявного поля по плоскости z = size_z/2."""
    lin_x = np.linspace(0.0, size_x, res_xy)
    lin_y = np.linspace(0.0, size_y, res_xy)
    
    cells_x = size_x / cell_size_mm
    cells_y = size_y / cell_size_mm
    
    x_ang = np.linspace(0, 2*np.pi*cells_x, res_xy)
    y_ang = np.linspace(0, 2*np.pi*cells_y, res_xy)
    X_ang, Y_ang = np.meshgrid(x_ang, y_ang, indexing="xy")
    Z_ang = np.zeros_like(X_ang)
    
    surf = LATTICE_SURFACES.get(lattice_type, LATTICE_SURFACES["gyroid"])
    field = surf["func"](X_ang, Y_ang, Z_ang)
    t, _ = compute_t_and_mean_grad(lattice_type, cell_size_mm, wall_mm)
    
    # маска: 1 = стенка, 0 = пустота
    mask = (np.abs(field) < t).astype(float)
    return lin_x, lin_y, mask


@st.cache_data
def estimate_fill_ratio(lattice_type, size_x, size_y, size_z, cell_size_mm, wall_mm, res=40):
    """Быстрая оценка коэффициента заполнения через выборку поля с broadcasting (низкое потребление RAM)."""
    cells_x = size_x / cell_size_mm
    cells_y = size_y / cell_size_mm
    cells_z = size_z / cell_size_mm
    
    xb = np.linspace(0, 2*np.pi*cells_x, res, endpoint=False, dtype=np.float32)[:, None, None]
    yb = np.linspace(0, 2*np.pi*cells_y, res, endpoint=False, dtype=np.float32)[None, :, None]
    zb = np.linspace(0, 2*np.pi*cells_z, res, endpoint=False, dtype=np.float32)[None, None, :]
    
    surf = LATTICE_SURFACES.get(lattice_type, LATTICE_SURFACES["gyroid"])
    F = surf["func"](xb, yb, zb)
    t, _ = compute_t_and_mean_grad(lattice_type, cell_size_mm, wall_mm)
    fill = float(np.mean(np.abs(F) < t))
    return fill


def filter_isolated_fragments(vol, min_ratio=0.01):
    """
    Удаляет мелкие несвязные осколки и уголки на границах блока,
    оставляя только цельное монолитное тело решетки.
    """
    solid_mask = (vol >= 0)
    labeled, num_features = ndimage.label(solid_mask)
    if num_features <= 1:
        return vol
    counts = np.bincount(labeled.flat)
    if len(counts) <= 1:
        return vol
    comp_counts = counts[1:]
    max_count = np.max(comp_counts)
    threshold = max_count * min_ratio
    keep_labels = set(np.where(counts >= threshold)[0])
    keep_labels.discard(0)
    remove_mask = solid_mask & (~np.isin(labeled, list(keep_labels)))
    vol[remove_mask] = -4.0
    return vol


@st.cache_data
def compute_preview_3d(lattice_type, size_x, size_y, size_z, cell_size_mm, wall_mm,
                       boundary_mode="open", model_type="solid", clean_islands=True):
    """Быстрый 3D-превью с низким потреблением RAM для плавного рендеринга."""
    cells_x = size_x / cell_size_mm
    cells_y = size_y / cell_size_mm
    cells_z = size_z / cell_size_mm
    
    max_cells = max(cells_x, cells_y, cells_z)
    preview_res = max(8, min(24, int(60 / max_cells)))
    
    Nx = max(8, int(round(preview_res * cells_x)))
    Ny = max(8, int(round(preview_res * cells_y)))
    Nz = max(8, int(round(preview_res * cells_z)))
    
    xb = np.linspace(0, 2 * np.pi * cells_x, Nx, endpoint=False, dtype=np.float32)[:, None, None]
    yb = np.linspace(0, 2 * np.pi * cells_y, Ny, endpoint=False, dtype=np.float32)[None, :, None]
    zb = np.linspace(0, 2 * np.pi * cells_z, Nz, endpoint=False, dtype=np.float32)[None, None, :]
    
    surf = LATTICE_SURFACES.get(lattice_type, LATTICE_SURFACES["gyroid"])
    F = surf["func"](xb, yb, zb)
    
    t, _ = compute_t_and_mean_grad(lattice_type, cell_size_mm, wall_mm)
    vol = t - np.abs(F, out=F)
    del xb, yb, zb, F
    
    vol = np.pad(
        vol,
        pad_width=1,
        mode="constant",
        constant_values=-10.0,
    )
    if clean_islands:
        vol = filter_isolated_fragments(vol)
    
    spacing = (cell_size_mm / preview_res,) * 3
    
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            vol,
            level=0.0,
            spacing=spacing,
            method="lewiner",
            allow_degenerate=False,
            step_size=1,
        )
        verts -= np.asarray(spacing)
        return verts, faces, normals
    except Exception:
        return None


@st.cache_data
def generate_lattice_stl(lattice_type, size_x, size_y, size_z, cell_size_mm, wall_mm, resolution,
                         boundary_mode="open", model_type="solid", clean_islands=True):
    """
    Генерирует высокоточный STL с ультра-низким потреблением RAM (оптимизировано под Streamlit Cloud 1GB).
    """
    cells_x = size_x / cell_size_mm
    cells_y = size_y / cell_size_mm
    cells_z = size_z / cell_size_mm

    Nx = int(round(resolution * cells_x))
    Ny = int(round(resolution * cells_y))
    Nz = int(round(resolution * cells_z))

    # Защита от OOM на облачном сервере (лимит RAM: 1 ГБ)
    total_voxels = Nx * Ny * Nz
    MAX_VOXELS = 8_000_000  # гарантирует RAM < 350 MB
    if total_voxels > MAX_VOXELS:
        scale_factor = (MAX_VOXELS / total_voxels) ** (1.0 / 3.0)
        Nx = max(8, int(Nx * scale_factor))
        Ny = max(8, int(Ny * scale_factor))
        Nz = max(8, int(Nz * scale_factor))

    # Broadcasting float32: 0 МБ на координатную сетку
    xb = np.linspace(0, 2 * np.pi * cells_x, Nx, endpoint=False, dtype=np.float32)[:, None, None]
    yb = np.linspace(0, 2 * np.pi * cells_y, Ny, endpoint=False, dtype=np.float32)[None, :, None]
    zb = np.linspace(0, 2 * np.pi * cells_z, Nz, endpoint=False, dtype=np.float32)[None, None, :]

    surf = LATTICE_SURFACES.get(lattice_type, LATTICE_SURFACES["gyroid"])
    F = surf["func"](xb, yb, zb)
    t, mean_grad = compute_t_and_mean_grad(lattice_type, cell_size_mm, wall_mm)

    vol = t - np.abs(F, out=F)
    del xb, yb, zb, F

    relative_density = float(np.mean(vol >= 0))

    vol = np.pad(
        vol,
        pad_width=1,
        mode="constant",
        constant_values=-10.0,
    )
    if clean_islands:
        vol = filter_isolated_fragments(vol)

    spacing = (cell_size_mm / resolution,) * 3

    verts, faces, normals, values = measure.marching_cubes(
        vol,
        level=0.0,
        spacing=spacing,
        method="lewiner",
        allow_degenerate=False,
        step_size=1,
    )

    verts -= np.asarray(spacing)
    del vol

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.merge_vertices()

    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)

    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    print(f"[{lattice_type.upper()}] t =", t)
    print("relative density =", relative_density)
    print("faces =", len(mesh.faces))
    print("watertight =", mesh.is_watertight)
    print("bounds, mm =", mesh.bounding_box.extents)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stl_filename = f"{lattice_type}_{int(size_x)}x{int(size_y)}x{int(size_z)}_c{int(cell_size_mm)}_w{wall_mm:.2f}.stl"
    stl_path = os.path.join(script_dir, "temp_model.stl")
    try:
        with open(stl_path, "ab"):
            pass
    except (PermissionError, OSError):
        stl_path = os.path.join(tempfile.gettempdir(), "temp_model.stl")

    stl_bytes = trimesh.exchange.stl.export_stl(mesh)
    with open(stl_path, "wb") as f:
        f.write(stl_bytes)

    return stl_path, resolution, len(mesh.faces), t, relative_density, mesh.is_watertight, mesh.bounding_box.extents, stl_filename


def create_3d_plot(verts, faces):
    """Строит интерактивный 3D-график с мягким Gouraud-шейдингом."""
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    ii, jj, kk = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh_3d = go.Mesh3d(
        x=x, y=y, z=z,
        i=ii, j=jj, k=kk,
        color='#5c88be',
        opacity=1.0,
        flatshading=False,
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.4, roughness=0.5, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo='skip'
    )

    fig = go.Figure(data=[mesh_3d])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=520
    )
    return fig


def create_slice_plot(lin_x, lin_y, mask, size_x, size_y):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=lin_x, y=lin_y, z=mask,
        colorscale=[[0, "rgba(20,24,36,1)"], [1, "#5c88be"]],
        showscale=False,
        hoverinfo='skip',
    ))
    # граница блока
    bx = [0, size_x, size_x, 0, 0]
    by = [0, 0, size_y, size_y, 0]
    fig.add_trace(go.Scatter(x=bx, y=by, mode='lines',
                             line=dict(color='#ff6b6b', width=1.5, dash='dash'),
                             hoverinfo='skip'))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, scaleanchor="y", scaleratio=1,
                   title=dict(text="X, мм", font=dict(size=10, color="#9ea4b0"))),
        yaxis=dict(showgrid=False, zeroline=False,
                   title=dict(text="Y, мм", font=dict(size=10, color="#9ea4b0"))),
        height=420
    )
    return fig


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ UI ---

def compact_input(label, min_v, max_v, default, step, key, help_text=""):
    c1, c2 = st.columns([1.1, 1.0])
    tooltip_attr = f'title="{help_text}"' if help_text else ""
    icon = " ⓘ" if help_text else ""
    c1.markdown(f'<div class="label-col" {tooltip_attr}>{label}{icon}</div>', unsafe_allow_html=True)
    return c2.number_input(label, min_v, max_v, default, step=step, key=key, label_visibility="collapsed")

def compact_selectbox(label, options, default_index, key, format_func=None, help_text=""):
    c1, c2 = st.columns([1.1, 1.0])
    tooltip_attr = f'title="{help_text}"' if help_text else ""
    icon = " ⓘ" if help_text else ""
    c1.markdown(f'<div class="label-col" {tooltip_attr}>{label}{icon}</div>', unsafe_allow_html=True)
    return c2.selectbox(label, options, index=default_index, key=key, format_func=format_func, label_visibility="collapsed")

def metric_card(label, value, unit="", tooltip=""):
    tooltip_attr = f' title="{tooltip}"' if tooltip else ''
    return (f'<div class="metric-box"{tooltip_attr}>'
            f'<div class="m-label">{label}</div>'
            f'<div class="m-value">{value}<span class="m-unit">{unit}</span></div>'
            f'</div>')


# ============================================================
# --- LAYOUT ---
# ============================================================

col_params, col_plot, col_metrics = st.columns([1.0, 3.0, 1.2])

# ============================================================
# --- ЛЕВАЯ КОЛОНКА (ПАРАМЕТРЫ) ---
# ============================================================
with col_params:
    st.markdown(f'<div class="section-header">{t("cell_params")}</div><div style="height: 12px;"></div>',
                unsafe_allow_html=True)

    # 1. Выбор типа решетки
    def format_lattice_name(k):
        return LATTICE_SURFACES[k]["name_" + st.session_state["lang"]]

    lattice_type = compact_selectbox(
        t("lattice_type_label"),
        LATTICE_KEYS,
        0,
        "lattice_type",
        format_func=format_lattice_name,
        help_text=t("lattice_type_help")
    )
    
    # Краткое описание физики выбранной структуры
    st.markdown(
        f'<div style="font-size:0.75rem; color:#7d8597; line-height:1.2; margin-top:-2px; margin-bottom:8px;">'
        f'{LATTICE_SURFACES[lattice_type]["desc_" + st.session_state["lang"]]}</div>',
        unsafe_allow_html=True
    )

    cell_size = compact_input(t("cell_size_label"), 1.0, 100.0, 10.0, 0.5,  "cell_size", t("cell_size_help"))
    wall_mm = compact_input(t("wall_label"),      0.1,  20.0,  1.20, 0.05, "wall_mm",   t("wall_help"))

    st.markdown(f'<div class="section-header">{t("model_params")}</div><div style="height: 12px;"></div>',
                unsafe_allow_html=True)

    size_x = compact_input(t("size_x_label"), 1.0, 5000.0, 20.0, 1.0, "size_x", t("size_x_help"))
    size_y = compact_input(t("size_y_label"), 1.0, 5000.0, 20.0, 1.0, "size_y", t("size_y_help"))
    size_z = compact_input(t("size_z_label"), 1.0, 5000.0, 20.0, 1.0, "size_z", t("size_z_help"))
    ro_v   = compact_input(t("ro_label"),    0.01,   20.0,  1.15, 0.01, "ro",   t("ro_help"))

    st.markdown(f'<div class="section-header">🎛️  {t("res_label")}</div><div style="height: 12px;"></div>', unsafe_allow_html=True)
    resolution = compact_input(t("res_label"), 10, 200, 80, 5, "resolution", t("res_help"))
    clean_islands = st.checkbox(t("clean_islands_label"), value=True, help=t("clean_islands_help"))

    # --- Валидация ---
    is_valid = True
    err_msg = ""
    if wall_mm >= cell_size / 2.0:
        is_valid = False
        err_msg = f"Толщина стенки ({wall_mm:.2f}) должна быть < cell_size/2 ({cell_size/2:.2f} мм)"

    if not is_valid:
        st.error(err_msg)

    # --- Хэш параметров для детекции изменений ---
    _param_hash = hash((lattice_type, cell_size, wall_mm, size_x, size_y, size_z, ro_v, resolution, clean_islands))

    st.write("")

    if st.button(t("btn_generate"), use_container_width=True, disabled=not is_valid):
        with st.spinner(t("generating")):
            try:
                stl_path, eff_res, num_faces, t_val, rel_density, is_watertight, extents, stl_filename = generate_lattice_stl(
                    lattice_type, size_x, size_y, size_z, cell_size, wall_mm, resolution, clean_islands=clean_islands
                )
                st.session_state['stl_ready_path'] = stl_path
                st.session_state['stl_filename']   = stl_filename
                st.session_state['stl_param_hash'] = _param_hash
                st.session_state['eff_res']        = eff_res
                st.session_state['num_faces']      = num_faces
                st.session_state['t_val']          = t_val
                st.session_state['rel_density']    = rel_density
                st.session_state['is_watertight']  = is_watertight
                st.session_state['extents']        = extents
            except Exception as e:
                st.error(f"Ошибка генерации STL: {e}")

    if 'stl_ready_path' in st.session_state:
        with open(st.session_state['stl_ready_path'], "rb") as f:
            st.download_button(
                t("btn_download"),
                f,
                st.session_state.get('stl_filename', "model.stl"),
                "application/sla",
                use_container_width=True
            )


# ============================================================
# --- ЦЕНТРАЛЬНАЯ КОЛОНКА (ВИЗУАЛИЗАЦИЯ) ---
# ============================================================
with col_plot:
    tab_slice_ui, tab_3d_ui = st.tabs([t("tab_slice"), t("tab_3d")])

    with tab_slice_ui:
        cur_name = LATTICE_SURFACES[lattice_type]["name_" + st.session_state["lang"]]
        st.markdown(f'<div class="section-header">{t("structure")} (2D Срез) — {cur_name}</div>', unsafe_allow_html=True)
        if is_valid:
            lin_x, lin_y, mask = compute_slice(lattice_type, size_x, size_y, cell_size, wall_mm)
            fig_slice = create_slice_plot(lin_x, lin_y, mask, size_x, size_y)
            st.plotly_chart(fig_slice, use_container_width=True, config={'displayModeBar': False})

            if st.session_state.get('stl_param_hash') == _param_hash and 'stl_ready_path' in st.session_state:
                st.info(f"✅ STL готов (Треугольников: {st.session_state.get('num_faces', 0):,})")
            elif 'stl_ready_path' in st.session_state:
                st.warning(t("3d_stale"))
        else:
            st.info("Исправьте параметры для просмотра среза.")

    with tab_3d_ui:
        cur_name = LATTICE_SURFACES[lattice_type]["name_" + st.session_state["lang"]]
        st.markdown(f'<div class="section-header">3D Визуализация — {cur_name}</div>', unsafe_allow_html=True)
        if is_valid:
            if st.session_state.get('stl_param_hash') == _param_hash and 'stl_ready_path' in st.session_state:
                with st.spinner("Отрисовка 3D модели..."):
                    preview_data = compute_preview_3d(lattice_type, size_x, size_y, size_z, cell_size, wall_mm, clean_islands=clean_islands)
                    if preview_data is not None:
                        verts_p, faces_p, _ = preview_data
                        fig_3d = create_3d_plot(verts_p, faces_p)
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.error("Данные 3D превью недоступны.")
            elif 'stl_ready_path' in st.session_state:
                st.warning(t("3d_stale"))
            else:
                st.info(t("3d_empty"))
        else:
            st.info("Исправьте параметры для просмотра 3D модели.")

# ============================================================
# --- ПРАВАЯ КОЛОНКА (МЕТРИКИ) ---
# ============================================================
with col_metrics:
    hdr_cols = st.columns([1.5, 2.0])
    with hdr_cols[0]:
        st.markdown(
            f'<div class="section-header" style="border-bottom:none; margin:0; padding:0; line-height:2.2;">'
            f'{t("metrics")}</div>',
            unsafe_allow_html=True
        )
    with hdr_cols[1]:
        lang_cols = st.columns([1.2, 1.0, 1.2])
        is_en = (st.session_state.get('lang', 'ru') == 'en')
        ru_color = "#ffffff" if not is_en else "#6f7380"
        en_color = "#ffffff" if is_en else "#6f7380"
        with lang_cols[0]:
            st.markdown(f'<div class="lang-label-ru" style="color: {ru_color};">рус</div>', unsafe_allow_html=True)
        with lang_cols[1]:
            st.toggle("Language", value=is_en, key="lang_toggle", label_visibility="collapsed")
        with lang_cols[2]:
            st.markdown(f'<div class="lang-label-en" style="color: {en_color};">EN</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-bottom: 1px solid #464b5d; margin-top: 2px; margin-bottom: 12px;"></div>',
                unsafe_allow_html=True)

    # --- Расчёт метрик ---
    cells_x = size_x / cell_size
    cells_y = size_y / cell_size
    cells_z = size_z / cell_size
    volume_full = size_x * size_y * size_z          # мм³
    t_val, _ = compute_t_and_mean_grad(lattice_type, cell_size, wall_mm)

    # Оценка заполнения (быстрая, через кэш)
    if is_valid:
        fill_ratio = estimate_fill_ratio(lattice_type, size_x, size_y, size_z, cell_size, wall_mm)
    else:
        fill_ratio = 0.0

    volume_mat = volume_full * fill_ratio            # мм³
    mass_g = volume_mat * ro_v * 1e-3               # г (мм³ * г/см³ * 0.001)

    tag_str = LATTICE_SURFACES[lattice_type]["tag"]

    metrics_list = [
        metric_card(t("m_size_x"),      f"{size_x:.0f}",       t("mm")),
        metric_card(t("m_size_y"),      f"{size_y:.0f}",       t("mm")),
        metric_card(t("m_size_z"),      f"{size_z:.0f}",       t("mm")),
        metric_card(t("m_cells_x"),     f"{cells_x:.1f}",      ""),
        metric_card(t("m_cells_y"),     f"{cells_y:.1f}",      ""),
        metric_card(t("m_cells_z"),     f"{cells_z:.1f}",      ""),
        metric_card(t("m_volume_full"), f"{volume_full:.0f}",  t("mm3")),
        metric_card(t("m_fill"),        f"{fill_ratio*100:.1f}", "%",
                    "Доля объёма, занятая материалом (оценка по полю)"),
        metric_card(t("m_mass"),        f"{mass_g:.2f}",       t("g")),
        metric_card(t("m_lattice"),     tag_str,               "", f"Тип TPMS структуры: {lattice_type}"),
        metric_card(t("m_threshold"),   f"{t_val:.3f}",        "",
                    "Порог |f| < t определяет толщину стенок. Рассчитан по градиенту."),
        metric_card(t("m_ro_mat"),      f"{ro_v:.2f}",         t("gcm3")),
    ]
    st.markdown('<div class="metrics-grid">' + "".join(metrics_list) + '</div>', unsafe_allow_html=True)

    # Дополнительные метрики из сгенерированного STL
    if 'stl_ready_path' in st.session_state and st.session_state.get('stl_param_hash') == _param_hash:
        st.markdown('<div class="section-header">📦 Свойства STL-модели</div><div style="height: 8px;"></div>', unsafe_allow_html=True)
        
        is_watertight = st.session_state.get('is_watertight', False)
        extents = st.session_state.get('extents', [0.0, 0.0, 0.0])
        rel_density = st.session_state.get('rel_density', 0.0)
        
        wt_str = "Да ✅" if is_watertight else "Нет ❌"
        
        stl_metrics = [
            metric_card("Watertight", wt_str, "", "Герметична ли сгенерированная STL-сетка"),
            metric_card("V-Fraction", f"{rel_density*100:.1f}", "%", "Относительная плотность сгенерированной сетки"),
            metric_card("Faces", f"{st.session_state.get('num_faces', 0):,}", "", "Число треугольников"),
        ]
        st.markdown('<div class="metrics-grid">' + "".join(stl_metrics) + '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="column-footer">'
        '© 2026 Demin A.I. — Laboratory of Mechanics of Novel Materials and Technologies IPMech RAS'
        '</div>',
        unsafe_allow_html=True
    )
