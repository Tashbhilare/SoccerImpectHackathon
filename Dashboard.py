import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from mplsoccer import Pitch

st.set_page_config(
    page_title="Bundesliga Valuation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .main { background: #0a0e1a; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0a0e1a 100%); }
    header[data-testid="stHeader"] {
        background: #0a0e1a !important;
        border-bottom: 1px solid #1e293b;
    }
    header[data-testid="stHeader"] * { color: #94a3b8 !important; }
    .stDeployButton, button[kind="header"] { color: #94a3b8 !important; }
    #MainMenu { visibility: hidden; }
    h1, h2, h3, h4 { color: #ffffff; font-weight: 700; }
    .hero-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 3rem 2rem; border-radius: 20px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(59,130,246,0.3);
    }
    .hero-title { font-size: 3.5rem; font-weight: 800; color: white; margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3); letter-spacing: -1px; }
    .hero-subtitle { font-size: 1.3rem; color: #e0f2fe; margin-top: 0.5rem; font-weight: 400; }
    .glass-card {
        background: rgba(30,41,59,0.6); backdrop-filter: blur(10px);
        border: 1px solid rgba(148,163,184,0.1); border-radius: 16px;
        padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }
    .glass-card:hover {
        transform: translateY(-8px); box-shadow: 0 16px 48px rgba(59,130,246,0.2);
        border-color: rgba(96,165,250,0.3);
    }
    .metric-value {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .metric-label {
        font-size: 0.9rem; color: #94a3b8; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 600; margin-top: 0.5rem;
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        margin: 2.5rem 0;
    }
    .section-title {
        font-size: 1.8rem; font-weight: 700; color: #60a5fa;
        margin: 2rem 0 1.5rem 0; padding-left: 1rem;
        border-left: 4px solid #3b82f6;
    }
    .player-badge {
        display: inline-block; padding: 0.5rem 1.2rem;
        border-radius: 25px; font-weight: 600; font-size: 0.9rem; margin: 0.5rem 0;
    }
    .badge-gem { background: linear-gradient(135deg,#10b981,#059669); color:white; box-shadow:0 4px 12px rgba(16,185,129,0.4); }
    .badge-overpriced { background: linear-gradient(135deg,#ef4444,#dc2626); color:white; box-shadow:0 4px 12px rgba(239,68,68,0.4); }
    .badge-elite { background: linear-gradient(135deg,#f59e0b,#d97706); color:white; box-shadow:0 4px 12px rgba(245,158,11,0.4); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #1e293b; padding: 0.5rem; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px;
        color: #94a3b8; font-weight: 600; padding: 0.8rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
    }
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stRadio label,
    div[data-testid="stWidgetLabel"] > label,
    div[data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    .stSelectbox > div > div {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 8px; color: #e2e8f0;
    }
    .stSlider > div > div > div { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    path = Path("data/processed/player_ratings_FINAL_with_market_values.parquet")
    if not path.exists():
        st.error("Data file not found. Expected: data/processed/player_ratings_FINAL_with_market_values.parquet")
        st.stop()
    df = pl.read_parquet(path).to_pandas()

    # Build player_id → team_name lookup and merge in
    try:
        pm     = pl.read_parquet("data/processed/player_metadata.parquet").to_pandas()
        squads = pl.read_parquet("data/processed/squads_metadata.parquet").to_pandas()
        # keep only the most-events team per player (handles dual-team edge cases)
        pm_top = (pm.dropna(subset=["player_id", "team_id"])
                    .sort_values("total_events", ascending=False)
                    .drop_duplicates(subset=["player_id"]))
        pm_top["team_id"]    = pm_top["team_id"].astype(str)
        squads["id"]         = squads["id"].astype(str)
        pm_top = pm_top.merge(squads.rename(columns={"id": "team_id", "name": "team_name"}),
                              on="team_id", how="left")
        df["player_id"] = df["player_id"].astype(str)
        pm_top["player_id"]  = pm_top["player_id"].astype(str)
        df = df.merge(pm_top[["player_id", "team_name"]], on="player_id", how="left")
    except Exception:
        df["team_name"] = "N/A"

    return df

@st.cache_data
def load_events():
    path = Path("data/processed/all_matches_with_zones.parquet")
    if path.exists():
        return pl.read_parquet(path).to_pandas()
    return None

try:
    player_data = load_data()
    all_events = load_events()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if "value_score" not in player_data.columns:
    player_data["value_score"] = (
        player_data["overall_rating"] / player_data["market_value_millions"].clip(lower=1)
    )
if "value_category" not in player_data.columns:
    def categorize(row):
        if pd.isna(row["market_value_millions"]):
            return "Unknown"
        r, v = row["overall_rating"], row["market_value_millions"]
        if r >= 80 and v < 15:   return "Hidden Gem"
        elif r >= 75 and v < 25: return "Great Value"
        elif r >= 85:            return "Elite"
        elif r < 65 and v >= 30: return "Overpriced"
        return "Fair Value"
    player_data["value_category"] = player_data.apply(categorize, axis=1)

RATING_COLS = [c for c in player_data.columns
               if c.endswith("_rating")
               and c not in ("overall_rating", "expected_rating")]
METRIC_LABELS = {c: c.replace("_rating", "").replace("_", " ").title() for c in RATING_COLS}

# ── Shared layout defaults ────────────────────────────────────────────────────

DARK_LAYOUT = dict(
    plot_bgcolor="#1e293b",
    paper_bgcolor="#0a0e1a",
    font=dict(color="#e2e8f0", family="Poppins", size=12),
    legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#475569",
                borderwidth=1, font=dict(color="#e2e8f0", size=13)),
)
XAXIS_DEFAULT = dict(gridcolor="#334155", zeroline=False)
YAXIS_DEFAULT = dict(gridcolor="#334155", zeroline=False)
YAXIS_NO_GRID = dict(gridcolor="rgba(0,0,0,0)", zeroline=False)

# ── Spatial toggle config ─────────────────────────────────────────────────────

SPATIAL_OPTIONS = {
    "forward":    ["Shot Map", "Carry / Dribble Zones", "Action Heatmap"],
    "midfielder": ["Progressive Passes", "Duel Locations", "Action Heatmap"],
    "defender":   ["Defensive Actions", "Progressive Passes", "Action Heatmap"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def dark_radar(player_row, metrics, title="", color="#60a5fa", fig_size=(9, 9)):
    vals = [player_row[m] for m in metrics if pd.notna(player_row.get(m))]
    lbls = [METRIC_LABELS.get(m, m) for m in metrics if pd.notna(player_row.get(m))]
    if not vals:
        return None
    n = len(vals)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#1e293b")
    v = vals + vals[:1]
    a = angles + angles[:1]
    ax.plot(a, v, "o-", linewidth=4, color=color, markersize=10)
    ax.fill(a, v, alpha=0.25, color=color)
    ax.set_xticks(angles)
    ax.set_xticklabels(lbls, size=11, color="#e2e8f0", weight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#64748b", size=10)
    ax.grid(True, color="#475569", linewidth=1.5, alpha=0.6)
    ax.spines["polar"].set_color("#64748b")
    if title:
        ax.set_title(title, size=13, color="white", pad=20, weight="bold")
    return fig


def coord_to_statsbomb(x, y):
    return (x + 52.5) * (120 / 105), (y + 34) * (80 / 68)


def draw_shot_map(ax, shots_df, pitch, player_name):
    goals = 0
    for _, s in shots_df.iterrows():
        sx, sy = coord_to_statsbomb(s["coordinates_x"], s["coordinates_y"])
        if s["result"] == "GOAL":
            pitch.scatter([sx], [sy], s=300, color="#10b981", edgecolors="white", linewidth=2.5, ax=ax, zorder=3)
            goals += 1
        elif s["result"] == "SAVED":
            pitch.scatter([sx], [sy], s=180, color="#fbbf24", edgecolors="white", linewidth=2, ax=ax, zorder=2)
        else:
            pitch.scatter([sx], [sy], s=100, color="#ef4444", edgecolors="white", linewidth=1.5, ax=ax, zorder=1, alpha=0.7)
    total = len(shots_df)
    conv = goals / total * 100 if total else 0
    ax.set_title(f"{player_name} — Shot Map\n{goals}G / {total}Sh | {conv:.0f}% Conv",
                 fontsize=12, fontweight="bold", color="white", pad=15)


def draw_action_heatmap(ax, events_df, pitch, player_name):
    """KDE heatmap of all player actions — universal across positions."""
    xs, ys = [], []
    for _, r in events_df.iterrows():
        if pd.notna(r.get("coordinates_x")) and pd.notna(r.get("coordinates_y")):
            x, y = coord_to_statsbomb(r["coordinates_x"], r["coordinates_y"])
            xs.append(x); ys.append(y)
    if not xs:
        ax.set_title(f"{player_name}\nNo data", color="white", fontsize=12)
        return
    pitch.kdeplot(xs, ys, ax=ax, fill=True, levels=100,
                  cmap="Blues", thresh=0.02, alpha=0.85)
    pitch.scatter(xs, ys, s=8, color="white", alpha=0.15, ax=ax)
    ax.set_title(f"{player_name} — Action Heatmap\n{len(xs)} actions",
                 fontsize=11, fontweight="bold", color="white", pad=12)


def draw_carry_zones(ax, events_df, pitch, player_name):
    """KDE heatmap of CARRY events — shows dribbling/progression zones."""
    carries = events_df[
        (events_df["event_type"] == "CARRY") &
        events_df["coordinates_x"].notna()
    ]
    if len(carries) == 0:
        ax.set_title(f"{player_name}\nNo carry data", color="white", fontsize=12)
        return
    xs = [coord_to_statsbomb(r["coordinates_x"], r["coordinates_y"])[0] for _, r in carries.iterrows()]
    ys = [coord_to_statsbomb(r["coordinates_x"], r["coordinates_y"])[1] for _, r in carries.iterrows()]
    pitch.kdeplot(xs, ys, ax=ax, fill=True, levels=80,
                  cmap="Oranges", thresh=0.05, alpha=0.85)
    pitch.scatter(xs, ys, s=10, color="white", alpha=0.2, ax=ax)
    ax.set_title(f"{player_name} — Carry / Dribble Zones\n{len(carries)} carries",
                 fontsize=11, fontweight="bold", color="white", pad=12)


def draw_defensive_actions(ax, events_df, pitch, player_name):
    """Scatter of DUEL, INTERCEPTION and CLEARANCE events by type."""
    def_types = ["DUEL", "INTERCEPTION", "CLEARANCE"]
    def_events = events_df[
        events_df["event_type"].isin(def_types) &
        events_df["coordinates_x"].notna()
    ]
    if len(def_events) == 0:
        ax.set_title(f"{player_name}\nNo defensive data", color="white", fontsize=12)
        return
    color_map = {"DUEL": "#f59e0b", "INTERCEPTION": "#10b981", "CLEARANCE": "#60a5fa"}
    for etype, grp in def_events.groupby("event_type"):
        xs, ys = zip(*[coord_to_statsbomb(r["coordinates_x"], r["coordinates_y"])
                       for _, r in grp.iterrows()])
        pitch.scatter(list(xs), list(ys), s=80,
                      color=color_map.get(etype, "#94a3b8"),
                      edgecolors="white", linewidth=0.8,
                      ax=ax, alpha=0.75, label=etype)
    ax.legend(loc="upper right", fontsize=9, facecolor="#1e293b",
              edgecolor="#475569", labelcolor="#e2e8f0")
    ax.set_title(f"{player_name} — Defensive Actions\n{len(def_events)} events",
                 fontsize=11, fontweight="bold", color="white", pad=12)


def draw_duel_locations(ax, events_df, pitch, player_name):
    """Scatter of DUEL events coloured by won/lost outcome."""
    duels = events_df[
        (events_df["event_type"] == "DUEL") &
        events_df["coordinates_x"].notna()
    ]
    if len(duels) == 0:
        ax.set_title(f"{player_name}\nNo duel data", color="white", fontsize=12)
        return
    won  = duels[duels["result"] == "WON"]
    lost = duels[duels["result"] != "WON"]
    for grp, clr, lbl in [(won, "#10b981", "Won"), (lost, "#ef4444", "Lost")]:
        if len(grp):
            xs, ys = zip(*[coord_to_statsbomb(r["coordinates_x"], r["coordinates_y"])
                           for _, r in grp.iterrows()])
            pitch.scatter(list(xs), list(ys), s=90, color=clr,
                          edgecolors="white", linewidth=0.8,
                          ax=ax, alpha=0.8, label=lbl)
    wr = len(won) / len(duels) * 100 if len(duels) else 0
    ax.legend(loc="upper right", fontsize=9, facecolor="#1e293b",
              edgecolor="#475569", labelcolor="#e2e8f0")
    ax.set_title(f"{player_name} — Duel Locations\n"
                 f"{len(won)}W / {len(lost)}L  ({wr:.0f}% win rate)",
                 fontsize=11, fontweight="bold", color="white", pad=12)


def draw_progressive_passes(ax, events_df, pitch, player_name):
    """Arrows for completed passes that advance the ball ≥10m forward."""
    prog = events_df[
        (events_df["event_type"] == "PASS") &
        events_df["coordinates_x"].notna() &
        events_df["end_coordinates_x"].notna()
    ]
    # support both 'result' string and 'success' bool columns
    if "result" in prog.columns:
        prog = prog[prog["result"] == "COMPLETE"]
    elif "success" in prog.columns:
        prog = prog[prog["success"] == True]
    prog = prog[prog["end_coordinates_x"] - prog["coordinates_x"] >= 10]
    if len(prog) == 0:
        ax.set_title(f"{player_name}\nNo progressive pass data", color="white", fontsize=12)
        return
    for _, r in prog.head(120).iterrows():
        x1, y1 = coord_to_statsbomb(r["coordinates_x"],     r["coordinates_y"])
        x2, y2 = coord_to_statsbomb(r["end_coordinates_x"], r["end_coordinates_y"])
        pitch.arrows(x1, y1, x2, y2, width=1.8, headwidth=5,
                     headlength=5, color="#3b82f6", alpha=0.5, ax=ax)
    ax.set_title(f"{player_name} — Progressive Passes\n{len(prog)} total",
                 fontsize=11, fontweight="bold", color="white", pad=12)


def render_event_breakdown(events_df):
    """Plotly horizontal bar of event type counts — used below pitch plots."""
    ev_counts = (
        events_df[~events_df["event_type"].str.startswith("GENERIC", na=False)]
        ["event_type"].value_counts()
        .reset_index()
    )
    ev_counts.columns = ["Event", "Count"]
    ev_counts = ev_counts.sort_values("Count", ascending=True).tail(6)
    fig = px.bar(ev_counts, x="Count", y="Event", orientation="h",
                 color="Count",
                 color_continuous_scale=["#1e3a8a", "#3b82f6", "#34d399"])
    fig.update_layout(
        plot_bgcolor="#1e293b", paper_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0", family="Poppins", size=11),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#334155", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
        margin=dict(l=5, r=5, t=5, b=5),
        height=220,
    )
    return fig


def render_spatial_plot(pev, player, player_name, view):
    """Renders the correct mplsoccer plot based on toggle selection."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1e293b",
                  line_color="#60a5fa", linewidth=2.5)
    fig_p, ax_p = pitch.draw(figsize=(9, 6))
    fig_p.patch.set_facecolor("#0a0e1a")

    if view == "Shot Map":
        shots = pev[(pev["event_type"] == "SHOT") & pev["coordinates_x"].notna()]
        draw_shot_map(ax_p, shots, pitch, player_name)

    elif view == "Carry / Dribble Zones":
        draw_carry_zones(ax_p, pev, pitch, player_name)

    elif view == "Defensive Actions":
        draw_defensive_actions(ax_p, pev, pitch, player_name)

    elif view == "Duel Locations":
        draw_duel_locations(ax_p, pev, pitch, player_name)

    elif view == "Progressive Passes":
        draw_progressive_passes(ax_p, pev, pitch, player_name)

    elif view == "Action Heatmap":
        clean = pev[~pev["event_type"].str.startswith("GENERIC", na=False)]
        draw_action_heatmap(ax_p, clean, pitch, player_name)

    return fig_p


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("### BUNDESLIGA\n### VALUATION SYSTEM")
st.sidebar.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#3b82f6,transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "", [" Home", " Player Search", " Value Targets",
         " Leaderboards", " Market Analysis", " Comparison"],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#3b82f6,transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)

gems_count = int(((player_data["overall_rating"] >= 80) &
                  (player_data["market_value_millions"] < 15) &
                  player_data["market_value_millions"].notna()).sum())

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    st.markdown(f'<div class="metric-label">PLAYERS</div><div class="metric-value">{len(player_data)}</div>', unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<div class="metric-label">VALUE TARGETS</div><div class="metric-value">{gems_count}</div>', unsafe_allow_html=True)


# ── HOME ──────────────────────────────────────────────────────────────────────
if page == " Home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">BUNDESLIGA VALUATION</div>
        <div class="hero-subtitle">Data-Driven Player Discovery for Competitive Advantage</div>
        <div style="margin-top:1rem;font-size:0.95rem;color:#bfdbfe;">2023/24 Season · 306 Matches · IMPECT Open Data</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("306",  "Matches Analyzed",  "Full 2023/24 Bundesliga Season"),
        ("963K", "Events Processed",  "Complete Event Coverage"),
        (str(gems_count), "Value Targets", "Elite Performance, Budget Price"),
        ("8",    "Metrics",           "Interpretable, Position-Specific"),
    ]
    for i, (val, lbl, desc) in enumerate(stats):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size:2.5rem;font-weight:800;
                    background:linear-gradient(135deg,#60a5fa,#34d399);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{val}</div>
                <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-top:0.5rem;">{lbl}</div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.3rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#60a5fa;margin-top:0;">Methodology</h4>
            <ul style="color:#cbd5e1;line-height:2.2;">
                <li><b>8 Interpretable Metrics</b> — Finishing, Chance Creation, Ball Progression,
                    Dribbling, Ball Winning, Defensive Actions, Passing Accuracy, Long Passing</li>
                <li><b>Within-Position Percentiles</b> — Forwards vs forwards, defenders vs defenders (0–99 scale)</li>
                <li><b>Role-Specific Weighting</b> — Forwards judged on finishing (45%), defenders on ball winning (35%)</li>
                <li><b>Market Integration</b> — Transfermarkt values for inefficiency detection</li>
                <li><b>Min. 500 minutes</b> played threshold for inclusion</li>
            </ul>
            <h4 style="color:#34d399;margin-top:1.5rem;">Key Innovation</h4>
            <p style="color:#cbd5e1;line-height:1.8;">
                Within-position comparison eliminates volume bias. Midfielders no longer dominate
                rankings simply because they touch the ball more.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Top Value Targets</div>', unsafe_allow_html=True)
        top_gems = player_data[
            (player_data["overall_rating"] >= 80) &
            (player_data["market_value_millions"] < 15) &
            (player_data["market_value_millions"].notna())
        ].nlargest(3, "overall_rating")
        for rank, (_, gem) in enumerate(top_gems.iterrows()):
            medals = ["🥇", "🥈", "🥉"]
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">
                            {medals[rank]} {gem['player_name']}
                        </div>
                        <div style="color:#94a3b8;margin-top:0.3rem;">
                            {gem['position'].title()} · {int(gem['matches_played'])} matches
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:2rem;font-weight:800;color:#34d399;">{gem['overall_rating']:.1f}</div>
                        <div style="color:#60a5fa;font-weight:600;">€{gem['market_value_millions']:.1f}M</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    ov1, ov2 = st.columns(2)

    with ov1:
        pos_dist = player_data["position"].value_counts().reset_index()
        pos_dist.columns = ["Position", "Count"]
        pos_dist["Position"] = pos_dist["Position"].str.title()
        fig_pos = px.bar(pos_dist, x="Position", y="Count", color="Position",
                         color_discrete_map={"Forward":"#ef4444","Midfielder":"#3b82f6","Defender":"#10b981"},
                         title="Players by Position")
        fig_pos.update_layout(**DARK_LAYOUT, showlegend=False, title_font_size=16)
        fig_pos.update_xaxes(**XAXIS_DEFAULT)
        fig_pos.update_yaxes(**YAXIS_DEFAULT)
        st.plotly_chart(fig_pos, use_container_width=True, config={"displayModeBar": False})

    with ov2:
        fig_hist = px.histogram(player_data, x="overall_rating", nbins=30,
                                color_discrete_sequence=["#3b82f6"], title="Overall Rating Distribution")
        fig_hist.update_layout(**DARK_LAYOUT, title_font_size=16)
        fig_hist.update_xaxes(**XAXIS_DEFAULT, title="Overall Rating")
        fig_hist.update_yaxes(**YAXIS_DEFAULT, title="# Players")
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ── PLAYER SEARCH ─────────────────────────────────────────────────────────────
elif page == " Player Search":
    st.markdown('<div class="section-title">Player Intelligence</div>', unsafe_allow_html=True)

    player_name = st.selectbox("Search Player Database", sorted(player_data["player_name"].dropna().unique()))
    player = player_data[player_data["player_name"] == player_name].iloc[0]

    # Role display — falls back gracefully if column missing
    role_display = player.get("role", "N/A")
    if pd.isna(role_display):
        role_display = "N/A"

    st.markdown(f"""
    <div class="hero-section" style="padding:2rem;margin-bottom:2rem;">
        <div style="font-size:2.5rem;font-weight:800;color:white;margin-bottom:0.5rem;">{player_name}</div>
        <div style="font-size:1.2rem;color:#e0f2fe;font-weight:500;">
            {player['position'].title()} · <span style="color:#fbbf24;">{role_display}</span> · {int(player['matches_played'])} Matches Played
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    mets = [
        ("Overall Rating",   f"{player['overall_rating']:.1f}", "/99", "#60a5fa"),
        ("Within Position",
         f"{player['within_position_overall']:.1f}" if pd.notna(player.get("within_position_overall")) else "N/A",
         "", "#34d399"),
        ("Market Value",
         f"€{player['market_value_millions']:.1f}M" if pd.notna(player.get("market_value_millions")) else "N/A",
         "", "#f59e0b"),
        ("Value Score",
         f"{player['value_score']:.2f}" if pd.notna(player.get("value_score")) else "N/A",
         "", "#a78bfa"),
    ]
    for i, (lbl, val, suf, clr) in enumerate(mets):
        with [c1, c2, c3, c4][i]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2.2rem;font-weight:800;color:{clr};">
                    {val}<span style="font-size:1.2rem;color:#64748b;">{suf}</span>
                </div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.5rem;
                    text-transform:uppercase;letter-spacing:1px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    if pd.notna(player.get("value_category")):
        cat = player["value_category"]
        # Display label maps — data values unchanged
        cat_display = cat.replace("Hidden Gem", "Value Target").replace("Overpriced", "Poor Value")
        bcls = ("badge-gem" if "Gem" in cat or "Great" in cat
                else "badge-overpriced" if "Overpriced" in cat else "badge-elite")
        st.markdown(f'<div class="player-badge {bcls}">{cat_display}</div>', unsafe_allow_html=True)

    # ── PLAYER STATS CARD ─────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="border-left-color:#a78bfa;">Player Profile & Stats</div>', unsafe_allow_html=True)

    pos_key_ps = player["position"].lower()

    # Team name — now merged in at load time
    team_display = player.get("team_name", "N/A")
    if pd.isna(team_display):
        team_display = "N/A"

    # ── Safe stat helpers ─────────────────────────────────────────────────────
    def stat_f(col):
        v = player.get(col)
        return round(float(v), 1) if pd.notna(v) else 0.0

    matches = stat_f("matches_played") or 1  # avoid div/0

    # ── Bio row ───────────────────────────────────────────────────────────────
    bio_c1, bio_c2, bio_c3 = st.columns(3)
    with bio_c1:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Club</div>
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-top:0.3rem;">🏟 {team_display}</div>
        </div>""", unsafe_allow_html=True)
    with bio_c2:
        role_display_bio = player.get("role", "N/A")
        if pd.isna(role_display_bio): role_display_bio = "N/A"
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Position / Role</div>
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-top:0.3rem;">⚽ {player['position'].title()} · <span style="color:#fbbf24;">{role_display_bio}</span></div>
        </div>""", unsafe_allow_html=True)
    with bio_c3:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Matches Played</div>
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-top:0.3rem;">📅 {int(matches)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Pull raw stat columns (now saved in enriched parquet) ─────────────────
    def stat_i(col):
        v = player.get(col)
        return int(v) if pd.notna(v) else 0

    goals           = stat_i("goals")
    shots           = stat_i("shots")
    assists         = stat_i("assists")
    duels_won       = stat_i("duels_won")
    duels_total     = stat_i("duels_total")
    interceptions   = stat_i("interceptions")
    recoveries      = stat_i("recoveries")
    carries         = stat_i("carries_completed")
    prog_passes     = stat_i("progressive_passes")
    passes_done     = stat_i("passes_completed")
    passes_total    = stat_i("passes_total")
    pass_pct        = stat_f("pass_completion_pct")
    duel_pct        = stat_f("duel_win_pct")
    conv_pct        = stat_f("shot_conversion_pct")

    # per-90 helpers derived from season totals ÷ matches
    def per90(n):
        return round(n / matches * 90, 2) if matches > 0 else 0.0

    # ── Position-specific stat grid ───────────────────────────────────────────
    if pos_key_ps == "forward":
        stat_items = [
            ("⚽ Goals",              str(goals),          "#34d399"),
            ("🎯 Shots",              str(shots),          "#60a5fa"),
            ("🅰️ Assists",            str(assists),        "#f59e0b"),
            ("📈 Conversion %",       f"{conv_pct}%",      "#a78bfa"),
            ("↗️ Progressive Passes", str(prog_passes),    "#38bdf8"),
            ("🏃 Carries Completed",  str(carries),        "#fb7185"),
        ]
        insight = (f"**{player_name}** scored **{goals} goals** from **{shots} shots** "
                   f"({conv_pct}% conversion) and provided **{assists} assists** across "
                   f"**{int(matches)} matches** — that's "
                   f"**{per90(goals):.2f} goals per 90**.")

    elif pos_key_ps == "midfielder":
        stat_items = [
            ("🅰️ Assists",            str(assists),        "#f59e0b"),
            ("↗️ Progressive Passes", str(prog_passes),    "#38bdf8"),
            ("✅ Pass Completion",    f"{pass_pct}%",      "#34d399"),
            ("🛡️ Duels Won",         f"{duels_won}/{duels_total}", "#60a5fa"),
            ("🔄 Recoveries",         str(recoveries),     "#a78bfa"),
            ("⚽ Goals",              str(goals),          "#fb7185"),
        ]
        insight = (f"**{player_name}** registered **{assists} assists** and "
                   f"**{prog_passes} progressive passes** with a "
                   f"**{pass_pct}% pass completion** rate, winning "
                   f"**{duels_won} of {duels_total} duels** over **{int(matches)} matches**.")

    else:  # defender
        stat_items = [
            ("🛡️ Interceptions",      str(interceptions),  "#34d399"),
            ("🔄 Recoveries",          str(recoveries),     "#60a5fa"),
            ("🤼 Duels Won",           f"{duels_won}/{duels_total}", "#f59e0b"),
            ("🤼 Duel Win %",          f"{duel_pct}%",      "#38bdf8"),
            ("✅ Pass Completion",     f"{pass_pct}%",      "#a78bfa"),
            ("↗️ Progressive Passes",  str(prog_passes),    "#fb7185"),
        ]
        insight = (f"**{player_name}** made **{interceptions} interceptions** and "
                   f"**{recoveries} recoveries**, winning **{duels_won} of {duels_total} "
                   f"duels** ({duel_pct}%) with a **{pass_pct}% pass completion** "
                   f"rate over **{int(matches)} matches**.")

    # Render stat grid (2 rows × 3 cols)
    for row_start in range(0, len(stat_items), 3):
        cols_st = st.columns(3)
        for ci, (lbl, val, clr) in enumerate(stat_items[row_start:row_start + 3]):
            with cols_st[ci]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center;padding:1rem;">
                    <div style="font-size:1.9rem;font-weight:800;color:{clr};">{val}</div>
                    <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:1px;margin-top:0.4rem;">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Natural language insight line
    st.markdown(f"""
    <div style="background:rgba(30,41,59,0.5);border-left:3px solid #3b82f6;
        padding:0.8rem 1.2rem;border-radius:0 8px 8px 0;margin-top:0.5rem;
        color:#cbd5e1;font-size:0.95rem;line-height:1.7;">
        💡 {insight}
    </div>""", unsafe_allow_html=True)

    # ── Per-90 bar chart using real raw counts ────────────────────────────────
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Build per-90 items relevant to position
    if pos_key_ps == "forward":
        p90_items = [
            ("Goals / 90",        per90(goals)),
            ("Assists / 90",      per90(assists)),
            ("Shots / 90",        per90(shots)),
            ("Prog. Passes / 90", per90(prog_passes)),
            ("Carries / 90",      per90(carries)),
        ]
    elif pos_key_ps == "midfielder":
        p90_items = [
            ("Assists / 90",      per90(assists)),
            ("Prog. Passes / 90", per90(prog_passes)),
            ("Recoveries / 90",   per90(recoveries)),
            ("Duels Won / 90",    per90(duels_won)),
            ("Goals / 90",        per90(goals)),
        ]
    else:
        p90_items = [
            ("Interceptions / 90", per90(interceptions)),
            ("Recoveries / 90",    per90(recoveries)),
            ("Duels Won / 90",     per90(duels_won)),
            ("Prog. Passes / 90",  per90(prog_passes)),
        ]

    p90_items = [(lbl, v) for lbl, v in p90_items if v > 0]
    if p90_items:
        p90_df = pd.DataFrame(p90_items, columns=["Metric", "Per 90"]) \
                   .sort_values("Per 90", ascending=True)
        fig_p90 = px.bar(
            p90_df, x="Per 90", y="Metric", orientation="h",
            color="Per 90",
            color_continuous_scale=["#1e3a8a", "#3b82f6", "#34d399"],
            title="Per-90 Averages (based on season totals)"
        )
        fig_p90.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="Poppins", size=11),
            coloraxis_showscale=False, title_font_size=14,
            xaxis=dict(gridcolor="#334155", zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
            margin=dict(l=5, r=5, t=35, b=5),
            height=260,
        )
        st.plotly_chart(fig_p90, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    viz1, viz2 = st.columns(2)

    with viz1:
        st.markdown('<div class="section-title" style="border-left-color:#34d399;">Performance Profile</div>', unsafe_allow_html=True)
        valid_cols = [c for c in RATING_COLS if pd.notna(player.get(c))]
        fig = dark_radar(player, valid_cols)
        if fig:
            st.pyplot(fig)
            plt.close()

    with viz2:
        st.markdown('<div class="section-title" style="border-left-color:#f59e0b;">Spatial Analysis</div>', unsafe_allow_html=True)

        pos_key = player["position"].lower()
        options = SPATIAL_OPTIONS.get(pos_key, ["Action Heatmap"])

        view = st.radio(
            "View",
            options,
            horizontal=True,
            label_visibility="collapsed",
            key="spatial_toggle"
        )

        if all_events is not None:
            pev = all_events[all_events["player_id"] == player["player_id"]]

            if len(pev) == 0:
                st.info("No event data found for this player.")
            else:
                fig_p = render_spatial_plot(pev, player, player_name, view)
                st.pyplot(fig_p)
                plt.close()

                # Plotly event breakdown bar beneath the pitch
                st.plotly_chart(
                    render_event_breakdown(pev),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
        else:
            st.info("Event data file not loaded.")

    # Metric breakdown bar
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="border-left-color:#a78bfa;">Metric Breakdown</div>', unsafe_allow_html=True)
    valid_cols = [c for c in RATING_COLS if pd.notna(player.get(c))]
    if valid_cols:
        metric_df = pd.DataFrame({
            "Metric": [METRIC_LABELS[c] for c in valid_cols],
            "Score":  [player[c] for c in valid_cols],
        }).sort_values("Score", ascending=True)
        fig_bar = px.bar(metric_df, x="Score", y="Metric", orientation="h",
                         color="Score", color_continuous_scale=["#1e3a8a","#3b82f6","#34d399"],
                         range_color=[0, 100])
        fig_bar.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="Poppins"),
            coloraxis_showscale=False,
            xaxis=dict(range=[0, 100], gridcolor="#334155"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=350, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# ── VALUE TARGETS ─────────────────────────────────────────────────────────────
elif page == " Value Targets":
    st.markdown('<div class="section-title">Value Target Discovery</div>', unsafe_allow_html=True)

    # Ordered by position group for better UX
    ROLE_ORDER = ["ST","CF","LW","RW","CAM","CM","CDM","LM","RM","CB","LB","RB","GK"]
    ALL_ROLES = [r for r in ROLE_ORDER
                 if r in player_data.get("role", pd.Series()).dropna().unique().tolist()
                 and r != "GK"] if "role" in player_data.columns else []

    # ── Filters — 2 row layout for breathing room ────────────────────────────
    filter_row1 = st.columns([1.2, 1.8])
    filter_row2 = st.columns(3)

    with filter_row1[0]:
        pos_f = st.multiselect(
            "Position",
            ["forward", "midfielder", "defender"],
            default=["forward", "midfielder", "defender"],
            format_func=lambda x: x.title()
        )
    with filter_row1[1]:
        role_f = st.multiselect(
            "Role  ",
            ALL_ROLES,
            default=ALL_ROLES,
            help="Filter by specific role — e.g. select only LW and RW to find wide forwards"
        )
    with filter_row2[0]:
        min_r = st.slider("Min Overall Rating", 0, 99, 75)
    with filter_row2[1]:
        max_p = st.slider("Max Market Value (€M)", 0, 80, 15)
    with filter_row2[2]:
        min_m = st.slider("Min Matches Played", 0, 34, 10)

    role_mask = (player_data["role"].isin(role_f)
                 if "role" in player_data.columns and role_f
                 else pd.Series([True] * len(player_data)))

    filtered = player_data[
        player_data["position"].isin(pos_f) &
        role_mask &
        (player_data["overall_rating"] >= min_r) &
        player_data["market_value_millions"].notna() &
        (player_data["market_value_millions"] <= max_p) &
        (player_data["market_value_millions"] > 0) &
        (player_data["matches_played"] >= min_m)
    ].sort_values("value_score", ascending=False)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # Active filter summary
    role_summary = ", ".join(role_f[:6]) + ("…" if len(role_f) > 6 else "") if role_f else "All roles"
    pos_summary  = ", ".join([p.title() for p in pos_f]) if pos_f else "All positions"
    st.markdown(f"""
    <div style="background:rgba(30,41,59,0.5);border-left:3px solid #3b82f6;
        padding:0.6rem 1rem;border-radius:0 8px 8px 0;margin-bottom:1rem;
        color:#94a3b8;font-size:0.85rem;">
        🔍 Filtering: <b style="color:#cbd5e1;">{pos_summary}</b> ·
        Roles: <b style="color:#cbd5e1;">{role_summary}</b> ·
        Rating ≥ <b style="color:#60a5fa;">{min_r}</b> ·
        Value ≤ <b style="color:#60a5fa;">€{max_p}M</b> ·
        Matches ≥ <b style="color:#60a5fa;">{min_m}</b>
    </div>
    """, unsafe_allow_html=True)

    # Results count as a proper metric card
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:1.5rem;">
        <div style="font-size:3rem;font-weight:800;
            background:linear-gradient(135deg,#60a5fa,#34d399);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            line-height:1;">{len(filtered)}</div>
        <div style="color:#94a3b8;font-size:0.9rem;text-transform:uppercase;
            letter-spacing:1px;font-weight:600;">players matching criteria</div>
    </div>
    """, unsafe_allow_html=True)

    if len(filtered) > 0:
        vt_role_col = ["role"] if "role" in filtered.columns else []
        disp = filtered[["player_name", "position"] + vt_role_col +
                        ["overall_rating", "market_value_millions",
                         "matches_played", "value_score"]].copy()
        disp.columns = (["Player", "Position"] + (["Role"] if vt_role_col else []) +
                        ["Rating", "Value (€M)", "Matches", "Value Score"])
        disp["Rating"]      = disp["Rating"].round(1)
        disp["Value (€M)"]  = disp["Value (€M)"].round(1)
        disp["Value Score"] = disp["Value Score"].round(2)
        disp["Position"]    = disp["Position"].str.title()
        st.dataframe(disp, use_container_width=True, height=400)

        # ── Gem drill-down: select a player from filtered list for spatial view ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.4rem;">Player Spotlight — Spatial View</div>', unsafe_allow_html=True)

        gem_names = filtered["player_name"].tolist()
        selected_gem = st.selectbox("Select a gem to inspect", gem_gems := gem_names, key="gem_spotlight")
        gem_row = player_data[player_data["player_name"] == selected_gem].iloc[0]
        gem_pos = gem_row["position"].lower()
        gem_options = SPATIAL_OPTIONS.get(gem_pos, ["Action Heatmap"])

        gem_view = st.radio(
            "Spatial view",
            gem_options,
            horizontal=True,
            label_visibility="collapsed",
            key="gem_spatial_toggle"
        )

        if all_events is not None:
            gem_ev = all_events[all_events["player_id"] == gem_row["player_id"]]
            if len(gem_ev) == 0:
                st.info("No event data for this player.")
            else:
                gcol1, gcol2 = st.columns([1.2, 1])
                with gcol1:
                    fig_gem_pitch = render_spatial_plot(gem_ev, gem_row, selected_gem, gem_view)
                    st.pyplot(fig_gem_pitch)
                    plt.close()
                with gcol2:
                    # Quick stat cards for the selected gem
                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom:1rem;">
                        <div style="font-size:1.4rem;font-weight:700;color:#34d399;">{selected_gem}</div>
                        <div style="color:#94a3b8;margin-top:0.3rem;">{gem_row['position'].title()}</div>
                        <hr style="border-color:#334155;margin:0.8rem 0;">
                        <div style="display:flex;justify-content:space-between;margin-top:0.5rem;">
                            <div>
                                <div style="font-size:1.8rem;font-weight:800;color:#60a5fa;">{gem_row['overall_rating']:.1f}</div>
                                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Rating</div>
                            </div>
                            <div>
                                <div style="font-size:1.8rem;font-weight:800;color:#f59e0b;">€{gem_row['market_value_millions']:.1f}M</div>
                                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Market Value</div>
                            </div>
                            <div>
                                <div style="font-size:1.8rem;font-weight:800;color:#a78bfa;">{gem_row['value_score']:.2f}</div>
                                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Value Score</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(
                        render_event_breakdown(gem_ev),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
        else:
            st.info("Event data file not loaded.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Value Score vs Rating</div>', unsafe_allow_html=True)
        color_col = "role" if "role" in filtered.columns else "position"
        fig_gem_scatter = px.scatter(
            filtered, x="market_value_millions", y="overall_rating",
            size="value_score", color=color_col, text="player_name",
            size_max=40,
            hover_data={"market_value_millions": ":.1f", "overall_rating": ":.1f",
                        "value_score": ":.2f", "matches_played": True,
                        "role": True if "role" in filtered.columns else False}
        )
        fig_gem_scatter.update_traces(textposition="top center", textfont=dict(size=9, color="white"))
        fig_gem_scatter.update_layout(**DARK_LAYOUT, height=500)
        fig_gem_scatter.update_xaxes(**XAXIS_DEFAULT, title="Market Value (€M)")
        fig_gem_scatter.update_yaxes(**YAXIS_DEFAULT, title="Overall Rating")
        st.plotly_chart(fig_gem_scatter, use_container_width=True, config={"displayModeBar": False})


# ── LEADERBOARDS ──────────────────────────────────────────────────────────────
elif page == " Leaderboards":
    st.markdown('<div class="section-title">Performance Rankings</div>', unsafe_allow_html=True)

    tab_overall, tab_metric, tab_position = st.tabs(["Overall", "By Metric", "By Position"])

    with tab_overall:
        top_n = st.slider("Display Top N", 10, 50, 20, key="lb_n")
        pos_filter = st.multiselect("Filter by Position",
                                    ["Forward","Midfielder","Defender"],
                                    default=["Forward","Midfielder","Defender"])
        mask = player_data["position"].str.title().isin(pos_filter)
        top  = player_data[mask].nlargest(top_n, "overall_rating")
        role_col = ["role"] if "role" in player_data.columns else []
        disp = top[["player_name", "position"] + role_col +
                   ["overall_rating", "market_value_millions", "matches_played"]].copy()
        disp.columns = (["Player", "Position"] + (["Role"] if role_col else []) +
                        ["Rating", "Value (€M)", "Matches"])
        disp["Rating"]     = disp["Rating"].round(1)
        disp["Value (€M)"] = disp["Value (€M)"].round(1)
        disp["Position"]   = disp["Position"].str.title()
        disp = disp.reset_index(drop=True); disp.index += 1
        st.dataframe(disp, use_container_width=True, height=600)

    with tab_metric:
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            metric_choice = st.selectbox("Select Metric", options=list(METRIC_LABELS.keys()),
                                         format_func=lambda x: METRIC_LABELS[x])
        with col_m2:
            top_n_m = st.slider("Top N", 5, 30, 15, key="lb_m_n")

        if metric_choice in player_data.columns:
            top_m = (player_data[player_data[metric_choice].notna()]
                     .nlargest(top_n_m, metric_choice)
                     [["player_name","position",metric_choice,"overall_rating","market_value_millions"]]
                     .copy())
            mlbl = METRIC_LABELS[metric_choice]
            top_m.columns = ["Player","Position",mlbl,"Overall","Value (€M)"]
            top_m["Position"]   = top_m["Position"].str.title()
            top_m[mlbl]         = top_m[mlbl].round(1)
            top_m["Overall"]    = top_m["Overall"].round(1)
            top_m["Value (€M)"] = top_m["Value (€M)"].round(1)
            top_m = top_m.reset_index(drop=True); top_m.index += 1

            fig_lb = px.bar(top_m.sort_values(mlbl), x=mlbl, y="Player",
                            orientation="h", color=mlbl,
                            color_continuous_scale=["#1e3a8a","#3b82f6","#34d399"],
                            range_color=[0,100], title=f"Top {top_n_m} by {mlbl}")
            fig_lb.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#0a0e1a",
                font=dict(color="#e2e8f0", family="Poppins"),
                coloraxis_showscale=False, title_font_size=16,
                xaxis=dict(range=[0,100], gridcolor="#334155"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                height=max(400, top_n_m * 28)
            )
            st.plotly_chart(fig_lb, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(top_m, use_container_width=True, height=400)

    with tab_position:
        pos_sel = st.selectbox("Position", ["Forward","Midfielder","Defender"], key="lb_pos")
        pos_data = player_data[player_data["position"].str.title() == pos_sel].nlargest(20, "overall_rating")
        avg_vals = {c: pos_data[c].mean() for c in RATING_COLS if c in pos_data.columns}

        lb_col1, lb_col2 = st.columns(2)
        with lb_col1:
            st.markdown(f"**Top 20 {pos_sel}s**")
            pos_disp = pos_data[["player_name","overall_rating","market_value_millions","matches_played"]].copy()
            pos_disp.columns = ["Player","Rating","Value (€M)","Matches"]
            pos_disp["Rating"]     = pos_disp["Rating"].round(1)
            pos_disp["Value (€M)"] = pos_disp["Value (€M)"].round(1)
            pos_disp = pos_disp.reset_index(drop=True); pos_disp.index += 1
            st.dataframe(pos_disp, use_container_width=True, height=500)
        with lb_col2:
            st.markdown(f"**Average Profile – Top 10 {pos_sel}s**")
            valid_avg = {c: v for c, v in avg_vals.items() if not np.isnan(v)}
            if valid_avg:
                avg_row = pd.Series(valid_avg)
                fig_avg = dark_radar(avg_row, list(valid_avg.keys()),
                                     title=f"Avg Top-10 {pos_sel}", color="#f59e0b", fig_size=(8,8))
                if fig_avg:
                    st.pyplot(fig_avg); plt.close()


# ── MARKET ANALYSIS ───────────────────────────────────────────────────────────
elif page == " Market Analysis":
    st.markdown('<div class="section-title">Market Inefficiency Analysis</div>', unsafe_allow_html=True)

    with_vals = player_data[player_data["market_value_millions"].notna()].copy()
    pos_colors = {"forward":"#ef4444","midfielder":"#3b82f6","defender":"#10b981"}

    fig = go.Figure()
    for pos in ["forward","midfielder","defender"]:
        d = with_vals[with_vals["position"] == pos]
        fig.add_trace(go.Scatter(
            x=d["market_value_millions"], y=d["overall_rating"],
            mode="markers", name=pos.title(),
            marker=dict(size=12, color=pos_colors[pos], opacity=0.8,
                        line=dict(width=1.5, color="white")),
            text=d["player_name"],
            customdata=np.stack([d["position"], d["overall_rating"],
                                 d["market_value_millions"], d["value_score"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>Position: %{customdata[0]}<br>"
                "Rating: %{customdata[1]:.1f}<br>Market Value: €%{customdata[2]:.1f}M<br>"
                "Value Score: %{customdata[3]:.2f}<extra></extra>"
            )
        ))
    for pos in ["forward","midfielder","defender"]:
        d = with_vals[with_vals["position"] == pos].dropna(
            subset=["market_value_millions","overall_rating"])
        if len(d) > 3:
            z = np.polyfit(d["market_value_millions"], d["overall_rating"], 1)
            xr = np.linspace(d["market_value_millions"].min(), d["market_value_millions"].max(), 50)
            fig.add_trace(go.Scatter(
                x=xr, y=np.polyval(z, xr), mode="lines",
                name=f"{pos.title()} trend",
                line=dict(color=pos_colors[pos], width=1.5, dash="dot"),
                showlegend=False, opacity=0.5
            ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Rating vs Market Value — All Players",
                   font=dict(size=22, color="#f1f5f9")),
        hovermode="closest", height=650
    )
    fig.update_xaxes(**XAXIS_DEFAULT, title="Market Value (€ Millions)")
    fig.update_yaxes(**YAXIS_DEFAULT, title="Overall Rating", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###  Value Targets\n*Rating ≥ 75 · Value < €15M*")
        gems = player_data[
            (player_data["overall_rating"] >= 75) &
            (player_data["market_value_millions"] < 15) &
            (player_data["market_value_millions"] > 0)
        ].nlargest(10, "overall_rating")[
            ["player_name","position","overall_rating","market_value_millions","value_score"]
        ].copy()
        gems.columns = ["Player","Position","Rating","Value (€M)","V-Score"]
        gems["Rating"]     = gems["Rating"].round(1)
        gems["Value (€M)"] = gems["Value (€M)"].round(1)
        gems["V-Score"]    = gems["V-Score"].round(2)
        gems["Position"]   = gems["Position"].str.title()
        st.dataframe(gems, use_container_width=True, height=380)

    with col2:
        st.markdown("###  Poor Value\n*Rating < 75 · Value ≥ €30M*")
        over = player_data[
            (player_data["overall_rating"] < 75) &
            (player_data["market_value_millions"] >= 30)
        ].nsmallest(10, "value_score")[
            ["player_name","position","overall_rating","market_value_millions","value_score"]
        ].copy()
        over.columns = ["Player","Position","Rating","Value (€M)","V-Score"]
        over["Rating"]     = over["Rating"].round(1)
        over["Value (€M)"] = over["Value (€M)"].round(1)
        over["V-Score"]    = over["V-Score"].round(2)
        over["Position"]   = over["Position"].str.title()
        st.dataframe(over, use_container_width=True, height=380)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Value Category Distribution</div>', unsafe_allow_html=True)
    vc = (player_data[player_data["market_value_millions"].notna()]
          ["value_category"].value_counts().reset_index())
    vc.columns = ["Category","Count"]
    fig_vc = px.pie(vc, names="Category", values="Count",
                    color_discrete_sequence=["#10b981","#3b82f6","#f59e0b","#ef4444","#a78bfa"])
    fig_vc.update_layout(
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0", family="Poppins"),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", font=dict(color="#e2e8f0", size=13)),
        height=400
    )
    st.plotly_chart(fig_vc, use_container_width=True, config={"displayModeBar": False})


# ── COMPARISON ────────────────────────────────────────────────────────────────
elif page == " Comparison":
    st.markdown('<div class="section-title">Head-to-Head Analysis</div>', unsafe_allow_html=True)

    all_names = sorted(player_data["player_name"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1: p1_name = st.selectbox("Player 1", all_names, key="p1")
    with col2: p2_name = st.selectbox("Player 2", all_names, index=1, key="p2")

    p1 = player_data[player_data["player_name"] == p1_name].iloc[0]
    p2 = player_data[player_data["player_name"] == p2_name].iloc[0]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, p, name, clr in [(c1,p1,p1_name,"#ef4444"),(c2,p2,p2_name,"#3b82f6")]:
        with col:
            val_str  = f"€{p['market_value_millions']:.1f}M" if pd.notna(p.get("market_value_millions")) else "N/A"
            role_str = p.get("role", ""); role_str = "" if pd.isna(role_str) else f" · {role_str}"
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;border-top:4px solid {clr};">
                <div style="font-size:1.8rem;font-weight:700;color:#f1f5f9;">{name}</div>
                <div style="color:#94a3b8;margin:0.5rem 0;">{p['position'].title()}<span style="color:#fbbf24;">{role_str}</span></div>
                <div style="font-size:2.5rem;font-weight:800;color:{clr};">{p['overall_rating']:.1f}</div>
                <div style="color:#60a5fa;font-size:1.1rem;font-weight:600;margin-top:0.5rem;">{val_str}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performance Radar</div>', unsafe_allow_html=True)

    shared = [c for c in RATING_COLS if pd.notna(p1.get(c)) and pd.notna(p2.get(c))]
    if shared:
        fig, ax = plt.subplots(figsize=(13, 11), subplot_kw=dict(projection="polar"))
        fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#1e293b")
        n = len(shared)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        a = angles + angles[:1]
        for p, name, clr in [(p1,p1_name,"#ef4444"),(p2,p2_name,"#3b82f6")]:
            v = [p[c] for c in shared] + [p[shared[0]]]
            ax.plot(a, v, "o-", linewidth=4, color=clr, label=name, markersize=10)
            ax.fill(a, v, alpha=0.2, color=clr)
        ax.set_xticks(angles)
        ax.set_xticklabels([METRIC_LABELS[c] for c in shared], size=12, color="#e2e8f0", weight="bold")
        ax.set_ylim(0,100); ax.set_yticks([25,50,75,100])
        ax.set_yticklabels(["25","50","75","100"], color="#94a3b8", size=11)
        ax.grid(True, color="#475569", linewidth=1.5, alpha=0.7)
        ax.spines["polar"].set_color("#64748b")
        leg = ax.legend(loc="upper right", fontsize=13, facecolor="#1e293b",
                        edgecolor="#475569", framealpha=0.9)
        for t in leg.get_texts(): t.set_color("#e2e8f0")
        st.pyplot(fig); plt.close()

    # Side-by-side grouped bar
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Metric Breakdown</div>', unsafe_allow_html=True)
    if shared:
        comp_df = pd.DataFrame({
            "Metric": [METRIC_LABELS[c] for c in shared] * 2,
            "Score":  [p1[c] for c in shared] + [p2[c] for c in shared],
            "Player": [p1_name]*len(shared) + [p2_name]*len(shared),
        })
        fig_cmp = px.bar(comp_df, x="Score", y="Metric", color="Player",
                         orientation="h", barmode="group",
                         color_discrete_map={p1_name:"#ef4444", p2_name:"#3b82f6"})
        fig_cmp.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="Poppins"),
            xaxis=dict(range=[0,100], gridcolor="#334155"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=400,
            legend=dict(bgcolor="rgba(30,41,59,0.8)", font=dict(color="#e2e8f0", size=13))
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    # ── Spatial comparison — works for ALL positions ──────────────────────────
    if all_events is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Spatial Comparison</div>', unsafe_allow_html=True)

        # Build shared toggle options: intersection of both players' position options
        p1_opts = SPATIAL_OPTIONS.get(p1["position"].lower(), ["Action Heatmap"])
        p2_opts = SPATIAL_OPTIONS.get(p2["position"].lower(), ["Action Heatmap"])
        # Always include Action Heatmap as universal fallback
        shared_opts = list(dict.fromkeys(p1_opts + p2_opts))  # union, preserving order

        cmp_view = st.radio(
            "Spatial view",
            shared_opts,
            horizontal=True,
            label_visibility="collapsed",
            key="cmp_spatial_toggle"
        )

        cmp_c1, cmp_c2 = st.columns(2)
        ev_breakdown_frames = []

        for col, p, name in [(cmp_c1, p1, p1_name), (cmp_c2, p2, p2_name)]:
            with col:
                pev = all_events[all_events["player_id"] == p["player_id"]]
                if len(pev) == 0:
                    st.info(f"No event data for {name}.")
                else:
                    pos_opts = SPATIAL_OPTIONS.get(p["position"].lower(), ["Action Heatmap"])
                    actual_view = cmp_view if cmp_view in pos_opts else "Action Heatmap"
                    fig_cmp_pitch = render_spatial_plot(pev, p, name, actual_view)
                    st.pyplot(fig_cmp_pitch)
                    plt.close()

                    # Collect event counts for combined chart below
                    ec = (
                        pev[~pev["event_type"].str.startswith("GENERIC", na=False)]
                        ["event_type"].value_counts()
                        .reset_index()
                    )
                    ec.columns = ["Event", "Count"]
                    ec["Player"] = name
                    ev_breakdown_frames.append(ec)

        # ── Combined event breakdown — shared scale ───────────────────────────
        if len(ev_breakdown_frames) == 2:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title" style="font-size:1.3rem;">Event Breakdown</div>',
                unsafe_allow_html=True
            )

            # Keep only top 6 event types by combined volume so chart stays readable
            combined = pd.concat(ev_breakdown_frames, ignore_index=True)
            top_events = (
                combined.groupby("Event")["Count"]
                .sum()
                .nlargest(6)
                .index.tolist()
            )
            combined = combined[combined["Event"].isin(top_events)]

            fig_ev_cmp = px.bar(
                combined,
                x="Count", y="Event", color="Player",
                orientation="h", barmode="group",
                color_discrete_map={p1_name: "#ef4444", p2_name: "#3b82f6"},
            )
            fig_ev_cmp.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#0a0e1a",
                font=dict(color="#e2e8f0", family="Poppins", size=12),
                legend=dict(
                    bgcolor="rgba(30,41,59,0.8)", bordercolor="#475569",
                    borderwidth=1, font=dict(color="#e2e8f0", size=13)
                ),
                xaxis=dict(gridcolor="#334155", zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
                margin=dict(l=5, r=5, t=10, b=5),
                height=300,
            )
            st.plotly_chart(fig_ev_cmp, use_container_width=True,
                            config={"displayModeBar": False})