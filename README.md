# 🤖 Robot Path Simulator

A* path planning simulator compared with Dijkstra, BFS and Greedy Best-First search.

**Live demo:** https://robot-path-simulator.streamlit.app

## Features
- Random obstacle maps with adjustable size and density
- Editable walls, start and goal positions
- Animated search and robot movement for each algorithm
- 4-way and 8-way (diagonal) movement
- A* vs other algorithms comparison with side-by-side maps, table and chart

## Algorithms
- **A\***: cost so far + estimated distance to goal (Manhattan for 4-way moves, Octile for diagonal), optimal and efficient
- **Dijkstra**: cost so far only, optimal but explores many cells
- **BFS**: level by level, ignores move cost
- **Greedy Best-First**: estimated distance only, fast but may give a longer path

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
