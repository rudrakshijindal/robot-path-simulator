import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathfinding import find_path, ALGORITHMS

st.set_page_config(page_title="Robot Path Simulator", page_icon="🤖", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h1 {background: linear-gradient(90deg,#00d2ff,#7b2ff7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Robot Path Simulator")
st.caption("A* path planning compared against Dijkstra, BFS and Greedy Best-First search.")

# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Environment")
size = st.sidebar.slider("Grid size", 10, 40, 20)
density = st.sidebar.slider("Obstacle density", 0.0, 0.4, 0.2, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

key = (size, density, seed)
if st.sidebar.button("🎲 Generate new map") or st.session_state.get("key") != key:
    rng = np.random.default_rng(int(seed))
    st.session_state.grid = (rng.random((size, size)) < density).astype(int)
    st.session_state.key = key

grid = st.session_state.grid

st.sidebar.header("📍 Start & Goal")
sr = st.sidebar.slider("Start row", 0, size - 1, 0)
sc = st.sidebar.slider("Start col", 0, size - 1, 0)
gr = st.sidebar.slider("Goal row", 0, size - 1, size - 1)
gc = st.sidebar.slider("Goal col", 0, size - 1, size - 1)

with st.sidebar.expander("🧱 Edit walls (rectangle)"):
    r1 = st.number_input("Row from", 0, size - 1, 0)
    r2 = st.number_input("Row to", 0, size - 1, 0)
    c1 = st.number_input("Col from", 0, size - 1, 0)
    c2 = st.number_input("Col to", 0, size - 1, 0)
    a, b = st.columns(2)
    if a.button("Add wall"):
        grid[min(r1, r2):max(r1, r2) + 1, min(c1, c2):max(c1, c2) + 1] = 1
    if b.button("Erase"):
        grid[min(r1, r2):max(r1, r2) + 1, min(c1, c2):max(c1, c2) + 1] = 0

grid[sr, sc] = 0
grid[gr, gc] = 0

st.sidebar.header("🧠 Algorithm")
algo = st.sidebar.selectbox("Choose algorithm", ALGORITHMS)
diagonal = st.sidebar.checkbox("Allow diagonal moves", value=False)
speed = st.sidebar.slider("Animation speed", 1, 10, 6)
run = st.sidebar.button("🚀 Run simulation", type="primary", width="stretch")

# ---------- Run the algorithm ----------
signature = (grid.tobytes(), sr, sc, gr, gc, algo, diagonal)
if run:
    st.session_state.result = find_path(grid, (sr, sc), (gr, gc), algo, diagonal)
    st.session_state.sig = signature

result = st.session_state.get("result") if st.session_state.get("sig") == signature else None


# ---------- Drawing ----------
def build_figure(grid, start, goal, result, speed):
    n = grid.shape[0]
    cell_px = max(6, int(420 / n))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=grid, colorscale=[[0, "#1b1f2e"], [1, "#6c7a96"]],
        showscale=False, xgap=2, ygap=2, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers", name="Explored",
        marker=dict(symbol="square", size=cell_px, color="#00b8d4", opacity=0.45)))
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="lines", name="Path",
        line=dict(color="#ffd600", width=4)))
    fig.add_trace(go.Scatter(
        x=[start[1]], y=[start[0]], mode="markers+text", name="Start", text=["S"],
        textfont=dict(color="black", size=12),
        marker=dict(size=22, color="#00e676", line=dict(width=2, color="white"))))
    fig.add_trace(go.Scatter(
        x=[goal[1]], y=[goal[0]], mode="markers+text", name="Goal", text=["G"],
        textfont=dict(color="white", size=12),
        marker=dict(size=22, color="#ff1744", line=dict(width=2, color="white"))))
    fig.add_trace(go.Scatter(
        x=[], y=[], mode="markers", name="Robot",
        marker=dict(size=20, color="#ffd600", symbol="diamond",
                    line=dict(width=2, color="white"))))

    buttons = []
    if result is not None and result["found"]:
        frames = []
        visited, path = result["visited"], result["path"]

        step = max(1, len(visited) // 100)
        for k in range(step, len(visited) + step, step):
            k = min(k, len(visited))
            frames.append(go.Frame(
                name=f"v{k}", traces=[1],
                data=[go.Scatter(x=[p[1] for p in visited[:k]],
                                 y=[p[0] for p in visited[:k]])]))

        for i in range(len(path)):
            frames.append(go.Frame(
                name=f"p{i}", traces=[2, 5],
                data=[go.Scatter(x=[p[1] for p in path[:i + 1]],
                                 y=[p[0] for p in path[:i + 1]]),
                      go.Scatter(x=[path[i][1]], y=[path[i][0]])]))

        fig.frames = frames
        duration = int(400 / speed)
        buttons = [
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=duration, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ]

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.5, n - 0.5], constrain="domain"),
        yaxis=dict(visible=False, range=[n - 0.5, -0.5], scaleanchor="x"),
        legend=dict(orientation="h", y=1.05, x=0),
        updatemenus=[dict(active=-1, type="buttons", direction="left", buttons=buttons,
                          x=0.0, xanchor="left", y=-0.04, yanchor="top",
                          bgcolor="#7b2ff7", bordercolor="#00d2ff",
                          font=dict(color="#111111", size=14),
                          pad=dict(l=6, r=6, t=4, b=4))] if buttons else [],
    )
    return fig


st.plotly_chart(build_figure(grid, (sr, sc), (gr, gc), result, speed), width="stretch")

# ---------- Stats ----------
if result is None:
    st.info("Pick an algorithm in the sidebar and press **🚀 Run simulation**, then press **▶ Play** below the grid.")
elif not result["found"]:
    st.error("No path exists. The robot is blocked. Try a lower obstacle density or erase some walls.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", algo)
    m2.metric("Cells explored", len(result["visited"]))
    m3.metric("Path length", result["length"])
    m4.metric("Compute time", f"{result['time_ms']} ms")

# ---------- Compare A* with other algorithms ----------
def mini_map(grid, start, goal, r, name):
    z = grid.astype(float).copy()
    for (a_, b_) in r["visited"]:
        z[a_, b_] = 2                      # explored cells
    fig = go.Figure(go.Heatmap(
        z=z, zmin=0, zmax=2, showscale=False, xgap=1, ygap=1, hoverinfo="skip",
        colorscale=[[0, "#1b1f2e"], [0.5, "#6c7a96"], [1, "#00b8d4"]]))
    if r["found"]:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in r["path"]], y=[p[0] for p in r["path"]],
            mode="lines", line=dict(color="#ffd600", width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=[start[1]], y=[start[0]], mode="markers",
                             marker=dict(size=11, color="#00e676"), showlegend=False))
    fig.add_trace(go.Scatter(x=[goal[1]], y=[goal[0]], mode="markers",
                             marker=dict(size=11, color="#ff1744"), showlegend=False))
    n = grid.shape[0]
    fig.update_layout(
        title=dict(text=f"<b>{name}</b><br><sup>{len(r['visited'])} cells explored · "
                        f"path {r['length']}</sup>", font=dict(size=15)),
        height=330, margin=dict(l=5, r=5, t=60, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.5, n - 0.5], constrain="domain"),
        yaxis=dict(visible=False, range=[n - 0.5, -0.5], scaleanchor="x"))
    return fig


st.divider()
st.subheader("📊 A* vs other algorithms")
st.caption("All four algorithms run on the same map. Cyan = cells the algorithm searched, "
           "yellow line = the path it chose. A* uses Manhattan distance for 4-way moves "
           "and Octile distance for diagonal moves.")

if st.button("⚖️ Compare A* with other algorithms"):
    results = {name: find_path(grid, (sr, sc), (gr, gc), name, diagonal)
               for name in ALGORITHMS}
    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm": name,
            "Path found": "Yes" if r["found"] else "No",
            "Path length": r["length"],
            "Cells explored": len(r["visited"]),
            "Time (ms)": r["time_ms"],
        })
    df = pd.DataFrame(rows)
    best = df["Path length"].min()
    df["Optimal?"] = df["Path length"].apply(lambda x: "✅" if x == best else "❌ longer")
    st.session_state.compare = df
    st.session_state.compare_results = results
    st.session_state.compare_start = ((sr, sc), (gr, gc))

if "compare" in st.session_state and "compare_results" in st.session_state:
    df = st.session_state.compare
    results = st.session_state.compare_results
    start, goal = st.session_state.compare_start

    # --- verdict banner ---
    a_star = results["A*"]
    if a_star["found"]:
        n_a = len(a_star["visited"])
        n_d = len(results["Dijkstra"]["visited"])
        n_b = len(results["BFS"]["visited"])
        saved = round((1 - n_a / max(n_d, n_b)) * 100)
        greedy = results["Greedy Best-First"]
        extra = ""
        if greedy["found"] and greedy["length"] > a_star["length"]:
            extra = (f" Greedy was fast but its path was longer "
                     f"({greedy['length']} vs {a_star['length']}).")
        st.success(f"🏆 A* found the shortest path ({a_star['length']}) while exploring "
                   f"**{saved}% fewer cells** than Dijkstra/BFS ({n_a} vs {max(n_d, n_b)}).{extra}")

    # --- four maps side by side ---
    cols = st.columns(4)
    for col, name in zip(cols, ALGORITHMS):
        col.plotly_chart(mini_map(grid, start, goal, results[name], name),
                         width="stretch")

    # --- table and bar chart ---
    left, right = st.columns([1, 1])
    left.dataframe(df, hide_index=True, width="stretch")

    colors = ["#ffd600" if a_ == "A*" else "#6c7a96" for a_ in df["Algorithm"]]
    fig = go.Figure(go.Bar(
        x=df["Algorithm"], y=df["Cells explored"], marker_color=colors,
        text=df["Cells explored"], textposition="outside"))
    fig.update_layout(
        title="Cells explored (lower = more efficient)",
        height=340, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#2a2f45"))
    right.plotly_chart(fig, width="stretch")

    st.caption("**A\\*** combines the cost so far with a distance estimate, so it finds the shortest "
               "path while exploring far fewer cells than **Dijkstra** and **BFS**. "
               "**Greedy Best-First** is fast but can return a longer path because it ignores the cost so far.")
