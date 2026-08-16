# OR-Tools Route Planner

[![Live Site](https://img.shields.io/badge/Live%20Site-or--tools--route--planner.vercel.app-2A9C64?style=for-the-badge&logo=vercel&logoColor=white)](https://or-tools-route-planner.vercel.app)

A vehicle routing web app that takes a set of store locations with delivery demand and generates optimised multi-vehicle delivery routes using Google OR-Tools. Built on synthetic Taipei road network data with a Leaflet map interface for visual route inspection and manual adjustment.

## What I did

Generated 50 store locations across central Taipei with daily demand profiles for each weekday, built driving time and distance matrices using the Haversine approximation with an urban road factor, and wired everything into a Flask backend that runs a constrained VRP solver. The frontend lets you load stores, drag-and-drop them between routes, rename drivers, and view per-route metrics in real time.

## Features

- Dijkstra-based shortest path routing via OR-Tools with configurable vehicle count, capacity, max stops, and max edge distance
- Adaptive solver time limit based on problem size
- Interactive Leaflet map with per-route colour coding and clickable store markers
- Drag-and-drop store reassignment between routes and available/dropped pools
- Real-time route metrics: travel time, distance, volume, and return leg

## Stack

Python, Flask, OR-Tools, pandas, Leaflet.js, Bootstrap

## Project structure

```
app.py                  — Flask backend and API routes
scheduling_algorithm.py — VRP model, solver, and route export
load_demand.py          — demand file loading and store merging
data/
  store_metadata.csv    — 50 Taipei store locations
  distance_matrix.csv   — pairwise driving distances
  duration_matrix.csv   — pairwise driving times
  demand/               — per-day demand CSVs (monday–friday)
interface.html          — single-page frontend
```
