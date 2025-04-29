# main.py  —  Canadian Federal-Election Night Simulator (Streamlit)
# -------------------------------------------------------------------------
#  Implements the final calibrated parameters supplied by the user.
# -------------------------------------------------------------------------

import random, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

# -------------------------------------------------------------------------
#  CONSTANTS  (calibrated)
# -------------------------------------------------------------------------
DATA_DIR          = Path("data")
SHAPEFILE_PATH    = DATA_DIR / "FED_CA_2021_EN.shp"
COLOUR_CSV_PATH   = DATA_DIR / "party_colours.csv"

# preference curve & geography
BETA        = 3.0
SIZE_SCALE  = 0.25

# province baselines  (will be re-zeroed after size modifier)
PROV_LEAN = {
    10: -0.10, 11: -0.05, 12: -0.05, 13: -0.05,
    24: -0.20,             35: -0.02,
    46:  0.20, 47:  0.30,  48:  0.40, 59:  0.10,
    60: -0.05, 61: -0.05,  62: -0.05,
}

# global default slider positions (must sum to 1.00)
DEFAULT_GLOBAL = {
    "Liberal":      0.30,
    "Conservative": 0.30,
    "NDP":          0.188,
    "Bloc":         0.08,
    "Green":        0.05,
    "PPC":          0.08,
    "Independent":  0.002,
}

# party positions on abstract left↔right axis
PARTY_IDEAL = {
    "NDP": -0.9, "Green": -0.8, "Liberal": -0.365, "Bloc": -0.25,
    "Independent": 0.0, "Conservative": 0.44, "PPC": 1.0,
}

# Bloc multipliers
BQ_BOOST_QC     = 3.0
BQ_SUPPRESS_ROC = 0.2

# Independent residual cap
IND_RESIDUAL_CAP = 0.15

# map / UI behaviour
SIMPLIFY_TOLERANCE = 0.009
BATCH              = 5         # ridings per animation frame
FRAME_DELAY        = 0.04      # seconds between frames
MAP_HEIGHT_PX      = 450
CAN_VIEW           = pdk.ViewState(latitude=56, longitude=-96, zoom=3.2, pitch=0)

# -------------------------------------------------------------------------
#  HELPER  – map renderer with tooltip
# -------------------------------------------------------------------------
def render_map(gdf, placeholder, pickable=False):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        get_fill_color="[fill_r, fill_g, fill_b, 150]",
        get_line_color=[255, 255, 255],
        pickable=pickable,
        auto_highlight=pickable,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=CAN_VIEW,
        map_style=None,
        tooltip={
            "html": "<b>{FEDNAME}</b><br/>Result: {winner}",
            "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white",
                      "fontSize": "12px"},
        },
    )
    placeholder.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT_PX)

# -------------------------------------------------------------------------
#  LOAD & PREP DATA
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(4326)

    gdf = (gdf.rename(columns={"ED_NAMEE": "FEDNAME"})
              [["FED_NUM", "FEDNAME", "geometry"]])

    gdf["PRCODE"] = (gdf["FED_NUM"] // 1000).astype(int)

    # urban-rural modifier
    proj     = gdf.to_crs(3347)
    log_area = np.log(proj.geometry.area)
    size_z   = (log_area - log_area.mean()) / log_area.std()
    size_mod = SIZE_SCALE * size_z

    # lean = province baseline + size; then centre to 0
    gdf["LEAN"] = gdf["PRCODE"].map(PROV_LEAN) + size_mod
    gdf["LEAN"] -= gdf["LEAN"].mean()

    # geometry simplification (no inward buffer)
    gdf["geometry"] = gdf.geometry.simplify(
        SIMPLIFY_TOLERANCE, preserve_topology=True
    )

    # colour placeholders & tooltip field
    colours = pd.read_csv(COLOUR_CSV_PATH, index_col="party")
    gdf[["fill_r", "fill_g", "fill_b"]] = 220     # light grey
    gdf["winner"] = "TBD"

    return gdf, colours


RIDINGS, COLOURS = load_data()
PARTIES          = list(DEFAULT_GLOBAL.keys())
N_RIDINGS        = len(RIDINGS)

# ─────────────────────────────── SIDEBAR ────────────────────────────────
st.sidebar.header("Global party probabilities")

# -- helper functions return unique keys ---------------------------------
def s_key(p): return f"s_{p}"
def n_key(p): return f"n_{p}"
def l_key(p): return f"l_{p}"

# 1️⃣  Normalise (happens BEFORE widgets are built)
if st.session_state.get("⚖️_normalize_now", False):
    locked_sum = sum(
        st.session_state.get(s_key(p), DEFAULT_GLOBAL[p])
        for p in PARTIES
        if st.session_state.get(l_key(p), False)
    )
    if locked_sum <= 1.0:
        unlocked = [p for p in PARTIES if not st.session_state.get(l_key(p), False)]
        unlocked_sum = sum(st.session_state.get(s_key(p), DEFAULT_GLOBAL[p]) for p in unlocked)
        if unlocked_sum:
            scale = (1 - locked_sum) / unlocked_sum
            for p in unlocked:
                new_val = st.session_state.get(s_key(p), DEFAULT_GLOBAL[p]) * scale
                st.session_state[s_key(p)] = new_val
                st.session_state[n_key(p)] = new_val
    st.session_state["⚖️_normalize_now"] = False

# 2️⃣  Seed keys once
for p in PARTIES:
    st.session_state.setdefault(s_key(p), DEFAULT_GLOBAL[p])
    st.session_state.setdefault(n_key(p), DEFAULT_GLOBAL[p])
    st.session_state.setdefault(l_key(p), False)

# 3️⃣  Build widgets
for p in PARTIES:

    def sync_from_slider(part=p):
        st.session_state[n_key(part)] = st.session_state[s_key(part)]

    def sync_from_number(part=p):
        st.session_state[s_key(part)] = st.session_state[n_key(part)]

    lock_col, slide_col, num_col = st.sidebar.columns([1, 4, 2], gap="small")

    # 🔒 checkbox  (unique label per party, hidden)
    with lock_col:
        st.checkbox(
            label=f"lock_{p}",             # non-empty, unique
            key=l_key(p),
            label_visibility="collapsed",
        )

    # slider  (Streamlit shows its label by default)
    with slide_col:
        st.slider(
            label=p,
            min_value=0.0, max_value=1.0, step=0.001,
            key=s_key(p),
            on_change=sync_from_slider,
        )

    # number-input  (unique dummy label, hidden)
    with num_col:
        st.number_input(
            label=f"num_{p}",              # non-empty, unique
            min_value=0.0, max_value=1.0,
            step=0.001, format="%.3f",
            key=n_key(p),
            on_change=sync_from_number,
            label_visibility="collapsed",
        )

# 4️⃣  Normalise button (sets flag, next run will rescale)
st.sidebar.button(
    "Normalize unlocked to 1.00",
    on_click=lambda: st.session_state.update({"⚖️_normalize_now": True})
)

# ── Reveal speed slider (10 ms – 250 ms) ────────────────────────────────
reveal_ms = st.sidebar.slider(
    "Reveal speed (milliseconds per update)",
    min_value=10, max_value=250, step=10, value=40,
    help="Lower = faster animation. Applies next time you press Run.",
)
reveal_delay = reveal_ms / 1000.0  # seconds

# 5️⃣  Collect final probabilities
global_probs = {p: st.session_state[s_key(p)] for p in PARTIES}

if abs(sum(global_probs.values()) - 1) > 1e-6:
    st.sidebar.warning("Total ≠ 1.00 (press normalize or adjust sliders).")

run_btn = st.sidebar.button("Run election night simulation", type="primary")
# ───────────────────────────── END SIDEBAR ──────────────────────────────

# -------------------------------------------------------------------------
#  UI LAYOUT
# -------------------------------------------------------------------------
map_col, table_col = st.columns([3, 1], gap="medium")
map_ph   = map_col.empty()
table_ph = table_col.empty()
render_map(RIDINGS, map_ph)          # initial centred grey map

# -------------------------------------------------------------------------
#  SIMULATION
# -------------------------------------------------------------------------
if run_btn:

    # ---------------- build probability matrix ---------------------------
    lean         = RIDINGS["LEAN"].values.reshape(-1, 1)            # N × 1
    party_vec    = np.array([PARTY_IDEAL[p] for p in PARTIES])      # 1 × P
    base_vec     = np.array([global_probs[p] for p in PARTIES])     # 1 × P

    weights = base_vec * np.exp(-BETA * (lean - party_vec) ** 2)

    # Bloc modifiers
    if "Bloc" in PARTIES:
        idx_bq = PARTIES.index("Bloc")
        in_qc  = RIDINGS["PRCODE"] == 24
        weights[in_qc, idx_bq]  *= BQ_BOOST_QC
        weights[~in_qc, idx_bq] *= BQ_SUPPRESS_ROC

    # Independent residual cap
    idx_ind = PARTIES.index("Independent")
    residual = weights.sum(axis=1) - weights[:, idx_ind]
    weights[:, idx_ind] = np.minimum(weights[:, idx_ind],
                                     residual * IND_RESIDUAL_CAP)

    weights = weights / weights.sum(axis=1, keepdims=True)

    # ---------------- east→west order with jitter ------------------------
    rng     = np.random.default_rng()
    jitter  = rng.normal(0, 0.6, N_RIDINGS)
    lon     = RIDINGS.to_crs(3347).centroid.to_crs(4326).x + jitter
    order   = RIDINGS.assign(lon=lon).sort_values("lon", ascending=False).index

    # ---------------- seat counter & animation --------------------------
    seat_counts = {p: 0 for p in PARTIES}

    for i, idx in enumerate(order, start=1):
        row   = RIDINGS.index.get_loc(idx)
        winner = rng.choice(PARTIES, p=weights[row])
        seat_counts[winner] += 1
        RIDINGS.loc[idx, "winner"] = winner
        RIDINGS.loc[idx, ["fill_r", "fill_g", "fill_b"]] = (
            COLOURS.loc[winner, ["r", "g", "b"]].values
        )

        if i % BATCH == 0 or i == N_RIDINGS:
            render_map(RIDINGS, map_ph, pickable=True)

            seat_df = (pd.Series(seat_counts, name="Seats")
                         .to_frame()
                         .sort_values("Seats", ascending=False))
            table_ph.dataframe(seat_df, height=MAP_HEIGHT_PX,
                               use_container_width=True)

            time.sleep(reveal_delay)
