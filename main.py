# app.py  —  Canadian Election-Night Simulator
# ---------------------------------------------------------------
#  1)  Global sliders set nation-wide support for each party.
#  2)  Per-riding odds are derived from a “left↔right lean”:
#        • Province baseline   (balances to mean ≈ 0)
#        • Riding size tweak   (urban → left, rural → right)
#  3)  Bloc Québécois gets an extra multiplier inside Québec.
#  4)  Ridings report Atlantic → Pacific with a sprinkle of noise.
#  5)  Geometry buffered inward to avoid colour spill into water.
# ---------------------------------------------------------------

import random, time
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk


# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------
SHAPEFILE_PATH      = "data/FED_CA_2021_EN.shp"
COLOUR_CSV_PATH     = "data/party_colours.csv"

SIMPLIFY_TOLERANCE  = 0.02      # geometry simplification (deg)
BUFFER_INWARD_DEG   = 0.03      # shrink polygons to keep fill on land
SIZE_SCALE          = 0.15      # strength of urban-rural modifier
BETA                = 2.0       # width of party preference curve
BQ_BOOST            = 3.0       # extra weight for Bloc in Quebec
BATCH               = 5         # ridings per redraw
FRAME_DELAY         = 0.04      # seconds between redraws
MAP_HEIGHT_PX       = 450

CAN_VIEW = pdk.ViewState(latitude=56, longitude=-96, zoom=3.2, pitch=0)


# ---------------------------------------------------------------------------
#  HELPER  – render map into placeholder
# ---------------------------------------------------------------------------
def render_map(gdf, placeholder, pickable=False):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gdf,
        get_fill_color="[fill_r, fill_g, fill_b, 150]",
        get_line_color=[255, 255, 255],
        pickable=pickable,
        auto_highlight=pickable,
    )

    tooltip = {
        "html": "<b>{FEDNAME}</b><br/>Result: {winner}",
        "style": {"backgroundColor": "rgba(0, 0, 0, 0.7)",
                  "color": "white",
                  "fontSize": "12px"}
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=CAN_VIEW,
        map_style=None,
        tooltip=tooltip,          #  ← NEW
    )
    placeholder.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT_PX)


# ---------------------------------------------------------------------------
#  DATA  – load shapefile, compute lean, prep colours
# ---------------------------------------------------------------------------
PROV_LEAN = {           # baseline; will be centred later
    10: -0.10, 11: -0.05, 12: -0.05, 13: -0.05,   # Atlantic
    24: -0.20,                                    # Quebec
    35: -0.05,                                    # Ontario
    46:  0.05, 47:  0.15, 48:  0.30,              # MB SK AB
    59:  0.00,                                    # BC
    60: -0.05, 61: -0.05, 62: -0.05,              # Territories
}

PARTY_IDEAL = {        # position on abstract left(-1)…right(+1) axis
    "NDP": -0.9,
    "Green": -0.8,
    "Liberal": -0.3,
    "Bloc": -0.25,
    "Independent": 0.0,
    "Conservative": 0.6,
    "PPC": 1.0,
}

@st.cache_data(show_spinner=True)
def load_data():
    """
    Loads the 2021 federal riding shapefile, adds province & lean data,
    and prepares colour/placeholder columns.

    Returns
    -------
    gdf : GeoDataFrame
        Riding polygons with extra columns:
        • PRCODE   – two-digit province/territory code
        • PRNAME   – short province string (NL, QC, ON, …)
        • LEAN     – continuous left(-) / right(+) score, mean≈0
        • fill_r/g/b – per-riding RGB, default light grey
    colours : DataFrame
        Party colour look-up (index = party, cols = r g b)
    std_area : float
        Standard deviation of log(area) in Lambert CRS (kept in case
        you want to tune SIZE_SCALE elsewhere).
    """
    # ---------------- read & basic housekeeping ---------------------------
    gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(4326)

    gdf = (gdf.rename(columns={"ED_NAMEE": "FEDNAME"})
              [["FED_NUM", "FEDNAME", "geometry"]])

    # province / territory codes & names
    gdf["PRCODE"] = (gdf["FED_NUM"] // 1000).astype(int)
    gdf["PRNAME"] = gdf["PRCODE"].map({
        10:"NL",11:"PE",12:"NS",13:"NB",24:"QC",35:"ON",
        46:"MB",47:"SK",48:"AB",59:"BC",60:"YT",61:"NT",62:"NU"
    })

    # ---------------- geometry simplification (no buffer) -----------------
    gdf["geometry"] = gdf.geometry.simplify(
        SIMPLIFY_TOLERANCE, preserve_topology=True
    )

    # ---------------- urban-rural size modifier --------------------------
    proj      = gdf.to_crs(3347)                       # StatsCan Lambert
    log_area  = np.log(proj.geometry.area)
    z         = (log_area - log_area.mean()) / log_area.std()
    size_mod  = -SIZE_SCALE * z                       # smaller = more left

    # ---------------- final LEAN  (province baseline + size) --------------
    gdf["LEAN"] = gdf["PRCODE"].map(PROV_LEAN) + size_mod
    gdf["LEAN"] = gdf["LEAN"] - gdf["LEAN"].mean()     # centre to 0

    # ---------------- colour placeholders ---------------------------------
    colours = pd.read_csv(COLOUR_CSV_PATH, index_col="party")

    for p in colours.index:
        gdf[p] = np.nan                                # future prob overrides

    gdf[["fill_r", "fill_g", "fill_b"]] = 220          # unrevealed = grey

    gdf["winner"] = "TBD"          # for the hover tooltip
    return gdf, colours, log_area.std()

RIDINGS, COLOURS, _ = load_data()
PARTIES = COLOURS.index.tolist()
N = len(RIDINGS)


# ---------------------------------------------------------------------------
#  SIDEBAR – global probabilities
# ---------------------------------------------------------------------------
st.sidebar.header("Global win probabilities (must sum to 1.00)")

default = dict(Liberal=0.30, Conservative=0.30, NDP=0.17,
               Bloc=0.07, Green=0.04, PPC=0.02, Independent=0.10)

global_probs = {
    p: st.sidebar.slider(p, 0.0, 1.0, default.get(p, 0.0), 0.01)
    for p in PARTIES
}
if abs(sum(global_probs.values()) - 1.0) > 1e-6:
    st.sidebar.error("Probabilities must add up to 1.00")
    st.stop()

run_btn = st.sidebar.button("Run election night simulation", type="primary")


# ---------------------------------------------------------------------------
#  LAYOUT  – map + seat table
# ---------------------------------------------------------------------------
map_col, table_col = st.columns([3, 1], gap="medium")
map_ph   = map_col.empty()
table_ph = table_col.empty()
render_map(RIDINGS, map_ph)                    # centred grey map


# ---------------------------------------------------------------------------
#  SIMULATION
# ---------------------------------------------------------------------------
if run_btn:
    # ---------------------------------------------------------------------
    # Pre-compute per-riding probability matrix  (N × P)
    # ---------------------------------------------------------------------
    lean = RIDINGS["LEAN"].values.reshape(-1, 1)             # N × 1
    party_ideal = np.array([PARTY_IDEAL[p] for p in PARTIES])  # 1 × P
    base = np.array([global_probs[p] for p in PARTIES])        # 1 × P

    weights = base * np.exp(-BETA * (lean - party_ideal)**2)   # broadcasting

    # Quebec boost for Bloc
    if "Bloc" in PARTIES:
        idx_bq = PARTIES.index("Bloc")
        weights[RIDINGS["PRCODE"] == 24, idx_bq] *= BQ_BOOST

    # Normalise rows
    weights = weights / weights.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------------------
    # Create east→west display order with Gaussian jitter
    # ---------------------------------------------------------------------
    rng    = np.random.default_rng()
    jitter = rng.normal(0, 2.5, N)
    centroids_lon = (
        RIDINGS.to_crs(3347).centroid.to_crs(4326).x       # accurate longitude
    )
    order = (
        RIDINGS.assign(lon=centroids_lon + jitter)
               .sort_values("lon", ascending=False)        # east → west
               .index
    )

    # Seat counter & simulation
    seat_counts = {p: 0 for p in PARTIES}

    for i, idx in enumerate(order, start=1):
        row = RIDINGS.index.get_loc(idx)
        winner = np.random.choice(PARTIES, p=weights[row])
        seat_counts[winner] += 1
        RIDINGS.loc[idx, "winner"] = winner          # ← NEW

        RIDINGS.loc[idx, ["fill_r", "fill_g", "fill_b"]] = (
            COLOURS.loc[winner, ["r", "g", "b"]].values
        )

        if i % BATCH == 0 or i == N:
            render_map(RIDINGS, map_ph, pickable=True)

            seat_df = (pd.Series(seat_counts, name="Seats")
                         .to_frame()
                         .sort_values("Seats", ascending=False))
            table_ph.dataframe(seat_df, height=MAP_HEIGHT_PX,
                               use_container_width=True)

            time.sleep(FRAME_DELAY)
