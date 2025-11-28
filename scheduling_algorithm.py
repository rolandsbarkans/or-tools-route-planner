import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
from folium import plugins


def process_store_data(store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a DataFrame containing store information into the format expected by VRP functions.
    
    Args:
        store_df: DataFrame containing store information with columns:
                 - store_id: Unique identifier for the store (will be converted to string)
                 - store_name: Name of the store
                 - address: Store address
                 - latitude: Latitude coordinate (float)
                 - longitude: Longitude coordinate (float)
                 - total_v: Volume/demand for the store (numeric, > 0)
                 
    Returns:
        pd.DataFrame: Processed DataFrame with consistent column names and data types
    """
    required_columns = ['store_id', 'store_name', 'address', 'latitude', 'longitude', 'total_v']
    
    # Create a copy of the dataframe
    df = store_df.copy()
    
    # Convert store_id to string
    df['store_id'] = df['store_id'].astype(str)
    
    # Ensure numeric types for coordinates and demand
    df['latitude'] = pd.to_numeric(df['latitude'], errors='raise')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='raise')
    df['total_v'] = pd.to_numeric(df['total_v'], errors='raise')
    
    return df[required_columns].copy()


def load_matrices(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    distance_matrix_path = data_dir / 'distance_matrix.csv'
    time_matrix_path = data_dir / 'duration_matrix.csv'

    def load_matrix(filepath: Path) -> pd.DataFrame:
        store_ids: List[str] = []
        values: List[List[float]] = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if not parts or not parts[0]:
                    continue
                sid = parts[0].strip()
                if not sid:
                    continue
                store_ids.append(str(sid))
                row_vals = [float(x) if x.strip() else 0.0 for x in parts[1:]]
                values.append(row_vals)
        if not store_ids:
            raise ValueError(f"Empty matrix: {filepath}")
        n = len(store_ids)
        for i in range(len(values)):
            while len(values[i]) < n:
                values[i].append(0.0)
            values[i] = values[i][:n]
        df = pd.DataFrame(values, index=store_ids, columns=store_ids, dtype=float)
        return df

    dist_df = load_matrix(distance_matrix_path)
    time_df = load_matrix(time_matrix_path)

    # Keep only stores present in both
    common = list(set(dist_df.index) & set(time_df.index))
    dist_df = dist_df.loc[common, common].astype(float)
    time_df = time_df.loc[common, common].astype(float)
    return dist_df, time_df


def _simple_kmeans(coords: Dict[str, Tuple[Optional[float], Optional[float]]], k: int, max_iters: int = 50, seed: int = 42) -> Dict[str, int]:
    """
    Lightweight K-Means for lat/lng to generate clusters without external deps.
    Returns labels dict: store_id -> cluster_id [0..k-1]. Stores without coords are excluded.
    """
    rng = __import__('random')
    points: List[Tuple[str, float, float]] = []
    for sid, (lat, lng) in coords.items():
        if lat is not None and lng is not None:
            points.append((sid, float(lat), float(lng)))
    if not points:
        return {}
    k = max(1, min(k, len(points)))
    rng.seed(seed)
    # init centers by random sample
    centers = rng.sample(points, k)
    centers_xy = [(c[1], c[2]) for c in centers]
    labels: Dict[str, int] = {}
    for _ in range(max_iters):
        # assign
        changed = False
        clusters: List[List[Tuple[float, float]]] = [[] for _ in range(k)]
        for sid, x, y in points:
            best_c = 0
            best_d = float('inf')
            for ci, (cx, cy) in enumerate(centers_xy):
                d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if d < best_d:
                    best_d = d
                    best_c = ci
            if labels.get(sid) != best_c:
                changed = True
            labels[sid] = best_c
            clusters[best_c].append((x, y))
        # update
        new_centers: List[Tuple[float, float]] = []
        for ci in range(k):
            if clusters[ci]:
                sx = sum(p[0] for p in clusters[ci])
                sy = sum(p[1] for p in clusters[ci])
                new_centers.append((sx / len(clusters[ci]), sy / len(clusters[ci])))
            else:
                # re-seed empty cluster
                sx, sy = centers_xy[ci]
                new_centers.append((sx, sy))
        centers_xy = new_centers
        if not changed:
            break
    return labels


def build_vrp_data(store_data: pd.DataFrame,
                   data_dir: Path,
                   num_vehicles: int,
                   vehicle_capacity: float,
                   max_route_time_minutes: int,
                   allow_drop: bool,
                   drop_penalty: int,
                   unloading_time_minutes: float,
                   cluster_k: Optional[int] = None,
                   cluster_penalty: int = 0,
                   balance_penalty: int = 50,
                   balance_span_coeff: int = 0,
                   max_edge_distance_m: Optional[float] = None,
                   max_stops_per_route: Optional[int] = None) -> Dict:
    """
    Build VRP data from store information.
    
    Args:
        store_data: DataFrame containing store information with columns:
                  - store_id: Unique identifier for the store (will be converted to string)
                  - store_name: Name of the store
                  - address: Store address
                  - latitude: Latitude coordinate (float)
                  - longitude: Longitude coordinate (float)
                  - total_v: Volume/demand for the store (numeric, > 0)
        data_dir: Path to the directory containing distance and time matrices
        num_vehicles: Number of vehicles available
        vehicle_capacity: Capacity of each vehicle
        max_route_time_minutes: Maximum route duration in minutes
        allow_drop: Whether to allow dropping stores
        drop_penalty: Penalty for dropping a store
        unloading_time_minutes: Time to unload at each store in minutes
        cluster_k: Number of clusters to create (defaults to num_vehicles)
        cluster_penalty: Penalty for inter-cluster travel
        balance_penalty: Penalty for unbalanced routes
        balance_span_coeff: Coefficient for balancing route spans
        max_edge_distance_m: Maximum allowed edge distance in meters
        max_stops_per_route: Maximum number of stops per route
        
    Returns:
        Dict containing VRP data
    """
    # Process the input store data
    stores_df = process_store_data(store_data)
    dist_df, time_df = load_matrices(data_dir)

    # Filter to stores that exist in matrices (exclude WH explicitly)
    store_ids = [str(sid) for sid in stores_df['store_id'] if str(sid) in time_df.index and str(sid) != 'WH']
    stores_df = stores_df[stores_df['store_id'].astype(str).isin(store_ids)].copy()

    # Create a dictionary mapping store_id to store information
    store_info = {}
    for _, row in stores_df.iterrows():
        store_id = str(row['store_id'])
        store_info[store_id] = {
            'store_name': str(row.get('store_name', store_id)),
            'lat': float(row['latitude']) if pd.notna(row['latitude']) else None,
            'lng': float(row['longitude']) if pd.notna(row['longitude']) else None,
            'address': str(row.get('address', ''))
        }

    # Node 0 is a dummy depot (no cost to/from)
    nodes = ['DEPOT'] + store_ids

    # Volumes may be fractional; scale to integers
    volume_scale = 10
    demands = [0] + [int(round(float(stores_df[stores_df['store_id'] == sid]['total_v'].iloc[0]) * volume_scale)) for sid in store_ids]
    unloading_times_seconds = [0] + [int(unloading_time_minutes * 60) for _ in store_ids]
    capacities = [int(round(vehicle_capacity * volume_scale)) for _ in range(num_vehicles)]

    # Time matrix in seconds; we only count store-to-store, depot arcs are zero
    n = len(nodes)
    time_matrix: List[List[int]] = [[0] * n for _ in range(n)]

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                time_matrix[i][j] = 0
            else:
                sid_i = nodes[i]
                sid_j = nodes[j]
                try:
                    t = time_df.loc[str(sid_i), str(sid_j)]
                except Exception:
                    t = 10**7  # very large to discourage
                time_matrix[i][j] = int(round(float(t)))

    # Distances
    distance_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                sid_i = nodes[i]
                sid_j = nodes[j]
                try:
                    d = dist_df.loc[str(sid_i), str(sid_j)]
                except Exception:
                    d = 0.0
                distance_matrix[i][j] = float(d)

    # Coordinates for map
    coords: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for _, row in stores_df.iterrows():
        lat = row['latitude'] if 'latitude' in row and pd.notna(row['latitude']) else None
        lng = row['longitude'] if 'longitude' in row and pd.notna(row['longitude']) else None
        coords[str(row['store_id'])] = (float(lat) if lat is not None else None,
                                        float(lng) if lng is not None else None)

    # Pre-cluster stores by coordinates to encourage spatial compactness
    if cluster_k is None:
        cluster_k = num_vehicles
    cluster_labels = _simple_kmeans(coords, k=cluster_k)

    max_route_time_seconds = int(max_route_time_minutes * 60)

    vrp_data = {
        'nodes': nodes,
        'total_demand': sum(demands)/volume_scale,
        'store_info': store_info,
        'num_vehicles': num_vehicles,
        'demands': demands,
        'capacities': capacities,
        'time_matrix': time_matrix,
        'distance_matrix': distance_matrix,
        'coords': coords,
        'store_ids': store_ids,
        'max_route_time_seconds': max_route_time_seconds,
        'allow_drop': allow_drop,
        'drop_penalty': int(drop_penalty),
        'cluster_labels': cluster_labels,
        'cluster_penalty': int(cluster_penalty),
        'balance_penalty': int(balance_penalty),
        'balance_span_coeff': int(balance_span_coeff),
        'max_edge_distance_m': float(max_edge_distance_m) if max_edge_distance_m is not None else None,
        'max_stops_per_route': int(max_stops_per_route) if max_stops_per_route is not None else None,
        'unloading_times_seconds': unloading_times_seconds,
    }
    
    return vrp_data


def solve_vrp(data: Dict, time_limit_seconds) -> Dict:
    nodes = data['nodes']
    num_vehicles = data['num_vehicles']
    n = len(nodes)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # If a max per-edge distance is provided, forbid store-to-store arcs exceeding it
    max_edge_m = data.get('max_edge_distance_m', None)
    forbidden_arcs = 0
    if max_edge_m is not None:
        for i in range(1, n):
            from_index = manager.NodeToIndex(i)
            for j in range(1, n):
                if i == j:
                    continue
                if data['distance_matrix'][i][j] > max_edge_m:
                    to_index = manager.NodeToIndex(j)
                    # Forbid i -> j hop if distance is too large
                    routing.solver().Add(routing.NextVar(from_index) != to_index)
                    forbidden_arcs += 1

    # Transit (time): only store-to-store counts; depot arcs = 0
    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == 0 or to_node == 0:
            return 0
        travel_time = int(data['time_matrix'][from_node][to_node])
        unloading_time = int(data['unloading_times_seconds'][to_node])
        return travel_time + unloading_time

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Objective cost = time + cluster crossing penalty (does not affect Time dimension)
    def cost_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == 0 or to_node == 0:
            return 0
        base = int(data['time_matrix'][from_node][to_node])
        penalty = 0
        if data.get('cluster_penalty', 0) > 0:
            labels: Dict[str, int] = data.get('cluster_labels', {})
            sid_from = nodes[from_node]
            sid_to = nodes[to_node]
            lf = labels.get(sid_from)
            lt = labels.get(sid_to)
            if lf is not None and lt is not None and lf != lt:
                penalty = int(data['cluster_penalty'])
        return base + penalty

    cost_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

    # Time dimension: limit per-vehicle work seconds (only between stops)
    routing.AddDimension(
        transit_callback_index,
        0,  # slack
        int(data['max_route_time_seconds']),
        True,
        'Time'
    )

    # Demand (volume) dimension
    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return int(data['demands'][from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [int(c) for c in data['capacities']],
        True,
        'Capacity'
    )

    # Visit count balancing dimension (each non-depot node contributes 1)
    def count_callback(from_index: int, to_index: int) -> int:
        to_node = manager.IndexToNode(to_index)
        return 1 if to_node != 0 else 0

    count_callback_index = routing.RegisterTransitCallback(count_callback)
    routing.AddDimension(
        count_callback_index,
        0,
        len(nodes),
        True,
        'Count'
    )
    count_dimension = routing.GetDimensionOrDie('Count')
    
    for v in range(num_vehicles):
        end_index = routing.End(v)
        # Hard cap: only user-defined maximum if provided; otherwise no explicit cap (use dimension default)
        user_cap = data.get('max_stops_per_route', None)
        if user_cap is not None:
            count_dimension.CumulVar(end_index).SetMax(int(user_cap))
        # Minimum stays at zero
        count_dimension.CumulVar(end_index).SetMin(0)

    # Remove balancing penalties by default when average cap is disabled; keep optional span if user set >0
    span_coeff = int(data.get('balance_span_coeff', 0))
    if span_coeff > 0:
        count_dimension.SetGlobalSpanCostCoefficient(span_coeff)

    # Allow dropping customers with penalty (to flag extremes)
    if data['allow_drop']:
        penalty = int(data['drop_penalty'])
        
        # Analyze typical travel times to see if penalty is reasonable
        sample_times = []
        for i in range(1, min(6, n)):  # Sample first 5 stores
            for j in range(1, min(6, n)):
                if i != j:
                    sample_times.append(data['time_matrix'][i][j])
        
    else:
        penalty = 10**9  # effectively forbid dropping

    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)
    search_parameters.log_search = False
    
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("No solution found under given constraints. Consider relaxing parameters.")

    # Extract routes
    routes: List[List[str]] = []
    route_times: List[int] = []
    route_distances: List[int] = []
    route_volumes: List[int] = []

    dropped: List[str] = []
    for node in range(1, n):
        index = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(index)) == index:
            dropped.append(nodes[node])

    for v in range(num_vehicles):
        index = routing.Start(v)
        vehicle_route: List[str] = []
        total_time = 0
        total_distance = 0
        total_volume = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            if node != 0:
                vehicle_route.append(nodes[node])
                total_volume += data['demands'][node]
            # accumulate time only when both are customers (store-to-store)
            if node != 0 and next_node != 0:
                travel_time = data['time_matrix'][node][next_node]
                travel_distance = data['distance_matrix'][node][next_node]
                unloading_time = data['unloading_times_seconds'][next_node]
                total_time += travel_time + unloading_time
                total_distance += travel_distance
            elif node == 0 and next_node != 0:
                unloading_time = data['unloading_times_seconds'][next_node]
                total_time += unloading_time
                travel_distance = data['distance_matrix'][node][next_node]
                total_distance += travel_distance
            index = next_index
        routes.append(vehicle_route)
        route_times.append(total_time)
        route_distances.append(total_distance)
        route_volumes.append(total_volume)

    # Convert volumes back to original scale
    volume_scale = 10
    route_volumes = [v / volume_scale for v in route_volumes]

    return {
        'routes': routes,
        'route_times_seconds': route_times,
        'route_distances_meters': route_distances,
        'route_volumes': route_volumes,
        'dropped_nodes': dropped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OR-Tools VRP (stores only, between-stops time)')
    parser.add_argument('--day', type=str, required=False, default='tuesday', help='monday|tuesday|...')
    parser.add_argument('--num-vehicles', type=int, required=True, help='Hard number of routes (vehicles)')
    parser.add_argument('--vehicle-capacity', type=float, required=True, help='Max volume capacity per vehicle')
    parser.add_argument('--max-route-time-minutes', type=int, required=True, help='Working duration per vehicle, minutes (between stops only)')
    parser.add_argument('--allow-drop', action='store_true', help='Allow dropping extreme stores')
    parser.add_argument('--drop-penalty', type=int, default=6000, help='Penalty cost for dropping a store (seconds)')
    parser.add_argument('--data-dir', type=str, default=str(Path(__file__).parent / 'data'))
    parser.add_argument('--time-limit', type=int, default=30, help='Solver time limit in seconds')
    parser.add_argument('--no-map', action='store_true', help='Disable map output')
    parser.add_argument('--clusters', type=int, default=None, help='Number of spatial clusters (default: = num-vehicles)')
    parser.add_argument('--cluster-penalty', type=int, default=0, help='Extra cost (seconds) when moving between different clusters')
    parser.add_argument('--balance-penalty', type=int, default=50, help='Soft penalty for exceeding target stops per vehicle')
    parser.add_argument('--balance-span', type=int, default=0, help='Global span cost on stop counts across vehicles (0 to disable)')
    parser.add_argument('--max-edge-distance-km', type=float, default=None, help='Maximum allowed distance (km) between consecutive stores; forbid hops exceeding this (store-to-store only)')
    parser.add_argument('--max-stops-per-route', type=int, default=None, help='Hard maximum number of stores per route (store-to-store count)')
    parser.add_argument('--unloading-time-minutes', type=float, default=None, help='Unloading time per store in minutes')
    return parser.parse_args()

def export_routes_for_ui(data: Dict, result: Dict, data_dir, unloading_time_minutes: float) -> Dict:
    """
    Export routes in a clean JSON format suitable for interactive UI.
    
    Args:
        data: Output from build_vrp_data function
        result: Output from solve_vrp function
    
    Returns:
        Dict containing structured data for UI consumption
    """
    
    # Extract basic data
    nodes = data['nodes']
    coords = data['coords']
    store_ids = data['store_ids']
    store_info = data['store_info']
    distance_mtrx, time_mtrx = load_matrices(data_dir)
    
    # Create store lookup for additional details
    store_lookup = {}
    for store_id in store_ids:
        store_lookup[store_id] = {
            'store_name': store_info.get(store_id, {}).get('store_name', store_id),
            'coordinates': coords.get(store_id, (None, None)),
            'address': store_info.get(store_id, {}).get('address', ''),
            'demand': data['demands'][nodes.index(store_id)] / 10,  # Convert back from scaled
            'unloading_time_minutes': data['unloading_times_seconds'][nodes.index(store_id)] / 60
        }
    
    # Process each route
    processed_routes = []
    for route_idx, route in enumerate(result['routes']):
        route_data = {
            'route_id': route_idx + 1,
            'stores': [],
            'metrics': {
                'total_stores': len(route),
                'total_delivery_time_minutes': result['route_times_seconds'][route_idx] / 60,
                'total_delivery_distance_kilometers': result['route_distances_meters'][route_idx] / 1000,
                'total_volume': result['route_volumes'][route_idx],
                'total_unloading_time_minutes': unloading_time_minutes * len(route),
                'total_route_time_minutes': 0,
                'total_route_distance_kilometers': 0
            },
            'constraints': {
                'max_capacity': data['capacities'][route_idx] / 10,  # Convert back from scaled
                'max_time_minutes': data['max_route_time_seconds'] / 60,
                'max_stops': data.get('max_stops_per_route')
            }
        }
        
        # Process each store in the route
        for stop_idx, store_id in enumerate(route):
            store_info = store_lookup[store_id]
            
            # Calculate travel time to this store (if not first stop)
            travel_time_minutes = 0
            travel_distance_kilometers = 0
            if stop_idx > 0:
                prev_store_id = route[stop_idx - 1]
                prev_node_idx = nodes.index(prev_store_id)
                curr_node_idx = nodes.index(store_id)
                travel_time_minutes = data['time_matrix'][prev_node_idx][curr_node_idx] / 60
                travel_distance_kilometers = data['distance_matrix'][prev_node_idx][curr_node_idx] / 1000
            else:
                travel_time_minutes = time_mtrx.loc["WH", store_id] / 60
                travel_distance_kilometers = distance_mtrx.loc["WH", store_id] / 1000
            
            store_data = {
                'store_id': store_id,
                'store_name': store_info.get('store_name', store_id),  # Fallback to store_id if name not available
                'store_address': store_info.get('address', ''),  # Empty string if address not available
                'stop_sequence': stop_idx + 1,
                'coordinates': {
                    'latitude': store_info['coordinates'][0],
                    'longitude': store_info['coordinates'][1]
                },
                'demand': store_info['demand'],
                'unloading_time_minutes': unloading_time_minutes,
                'travel_time_to_store_minutes': travel_time_minutes,
                'travel_distance_kilometers': travel_distance_kilometers,
                'cumulative_time_minutes': 0,  # Will calculate below
                'cumulative_distance_kilometers': 0,  # Will calculate below
                'cumulative_volume': 0  # Will calculate below
            }
            
            route_data['stores'].append(store_data)
        
        warehouse_info = {
            'store_id': "WH",
            'store_name': "Warehouse",
            'store_address': "Warehouse Address",
            'stop_sequence': 0,
            'coordinates': {
                'latitude': 56.924874436201804,
                'longitude': 23.981047131295707
            },
            'demand': 0,
            'unloading_time_minutes': 0,
            'travel_time_to_store_minutes': 0,
            'travel_distance_kilometers': 0,
            'cumulative_time_minutes': 0,
            'cumulative_distance_kilometers': 0,
            'cumulative_volume': 0
        }

        warehouse_start = warehouse_info.copy()
        route_data['stores'].insert(0, warehouse_start)

        warehouse_end = warehouse_info.copy()
        warehouse_end['stop_sequence'] = len(route_data['stores'])
        if route_data['stores'] and len(route_data['stores']) > 1:
            last_store = route_data['stores'][-1]
            last_store_id = last_store['store_id']
            warehouse_id = "WH"
            warehouse_end['travel_time_to_store_minutes'] = time_mtrx.loc[last_store_id, warehouse_id] / 60
            warehouse_end['travel_distance_kilometers'] = distance_mtrx.loc[last_store_id, warehouse_id] / 1000
        route_data['stores'].append(warehouse_end)

        for i, store in enumerate(route_data['stores']):
            store['stop_sequence'] = i

        # Calculate cumulative metrics
        cumulative_time = 0
        cumulative_distance = 0
        cumulative_volume = 0
        for store_data in route_data['stores']:
            cumulative_time += store_data['travel_time_to_store_minutes'] + store_data['unloading_time_minutes']
            cumulative_distance += store_data['travel_distance_kilometers']
            cumulative_volume += store_data['demand']
            store_data['cumulative_time_minutes'] = cumulative_time
            store_data['cumulative_distance_kilometers'] = cumulative_distance
            store_data['cumulative_volume'] = cumulative_volume
        
        if route:  # Only calculate if route has stores
            # Last store to warehouse
            last_store_id = route[-1]
            # Use the full matrices (with WH row/col) to capture the true return leg
            last_to_warehouse_time = time_mtrx.loc[last_store_id, "WH"] / 60
            last_to_warehouse_distance = distance_mtrx.loc[last_store_id, "WH"] / 1000

            route_data['metrics']['total_route_time_minutes'] = cumulative_time + last_to_warehouse_time
            route_data['metrics']['total_route_distance_kilometers'] = cumulative_distance + last_to_warehouse_distance

        processed_routes.append(route_data)
    
    # Process dropped stores
    dropped_stores = []
    for store_id in result['dropped_nodes']:
        store_info = store_lookup.get(store_id, {})
        
        # Get store name with fallback logic
        store_name = store_info.get('store_name')
        if not store_name or pd.isna(store_name) or store_name == store_id:
            store_name = store_info.get('name', store_id)
            
        dropped_store = {
            'store_id': store_id,
            'store_name': store_info.get('store_name', store_id),  # Fallback to store_id if name not available
            'store_address': store_info.get('address', ''),  # Empty string if address not available
            'coordinates': {
                'latitude': store_info.get('coordinates', (None, None))[0],
                'longitude': store_info.get('coordinates', (None, None))[1]
            },
            'demand': store_info['demand'],
            'unloading_time_minutes': store_info['unloading_time_minutes'],
            'reason': 'constraint_violation'  # Could be enhanced with specific reasons
        }
        dropped_stores.append(dropped_store)
    
    # Create summary statistics
    total_metrics = {
        'total_vehicles': len(processed_routes),
        'total_stores_served': sum(len(route['stores']) for route in processed_routes) - 2 * len(processed_routes),
        'total_stores_dropped': len(dropped_stores),
        'total_volume_delivered': sum(route['metrics']['total_volume'] for route in processed_routes),
        'total_delivery_time_minutes': (sum(route['metrics']['total_delivery_time_minutes'] for route in processed_routes)),
        'total_delivery_distance_kilometers': int(sum(route['metrics']['total_delivery_distance_kilometers'] for route in processed_routes)),
        'total_route_time_minutes': int(sum(route['metrics']['total_route_time_minutes'] for route in processed_routes)),
        'total_route_distance_kilometers': int(sum(route['metrics']['total_route_distance_kilometers'] for route in processed_routes)),
        'average_delivery_time_minutes': sum(route['metrics']['total_delivery_time_minutes'] for route in processed_routes) / len(processed_routes) if processed_routes else 0,
        'average_delivery_distance_kilometers': sum(route['metrics']['total_delivery_distance_kilometers'] for route in processed_routes) / len(processed_routes) if processed_routes else 0,
        'average_route_time_minutes': sum(route['metrics']['total_route_time_minutes'] for route in processed_routes) / len(processed_routes) if processed_routes else 0,
        'average_route_distance_kilometers': sum(route['metrics']['total_route_distance_kilometers'] for route in processed_routes) / len(processed_routes) if processed_routes else 0,
        'average_route_volume': sum(route['metrics']['total_volume'] for route in processed_routes) / len(processed_routes) if processed_routes else 0
    }
    
    # Create constraint summary
    constraint_summary = {
        'max_route_time_minutes': data['max_route_time_seconds'] / 60,
        'vehicle_capacity': data['capacities'][0] / 10 if data['capacities'] else 0,  # Assuming all vehicles same capacity
        'max_stops_per_route': data.get('max_stops_per_route'),
        'max_edge_distance_km': data.get('max_edge_distance_m', 0) / 1000 if data.get('max_edge_distance_m') else None,
        'cluster_settings': {
            'num_clusters': len(set(data.get('cluster_labels', {}).values())),
            'cluster_penalty': data.get('cluster_penalty', 0),
            'balance_span_coeff': data.get('balance_span_coeff', 0)
        }
    }
    
    # Final output structure
    ui_data = {
        'summary': total_metrics,
        'constraints': constraint_summary,
        'routes': processed_routes,
        'dropped_stores': dropped_stores,
    }
    
    return ui_data


def get_store_distances_times(store_ids: List[str], data_dir: Path) -> List[Dict]:
    """
    Retrieve time and distance information between stores using the distance and time matrices.
    
    Args:
        store_ids: List of store IDs (as strings) to get information for
        data_dir: Path to the directory containing the matrix files
        
    Returns:
        List of dictionaries containing store-to-store travel information:
        [
            {
                'from_store': 'store_id_1',  # string
                'to_store': 'store_id_2',    # string
                'distance_km': float,        # in kilometers, rounded to 2 decimal places
                'time_minutes': float        # in minutes, rounded to 1 decimal place
            },
            ...
        ]
    """
    # Ensure all store IDs are strings to match matrix indices
    store_ids = [str(store_id) for store_id in store_ids]
    
    # Load the full matrices using the existing load_matrices function
    dist_df, time_df = load_matrices(data_dir)
    
    # Get the intersection of requested store IDs and available matrix indices
    available_stores = set(dist_df.index) & set(time_df.index) & set(store_ids)
    
    # Filter the dataframes to only include the requested store IDs
    dist_df = dist_df.loc[list(available_stores), list(available_stores)]
    time_df = time_df.loc[list(available_stores), list(available_stores)]
    
    # Generate all pairs of stores
    results = []
    for from_store in available_stores:
        for to_store in available_stores:
            if from_store == to_store:
                continue  # Skip self-referential distances
            
            # Get distance and time values directly from DataFrames using .loc
            distance = dist_df.loc[from_store, to_store]
            time_val = time_df.loc[from_store, to_store]
            
            # Ensure values are floats and apply scaling if needed
            try:
                # Convert to float and handle None/NaN values
                distance = float(distance) if distance is not None and not pd.isna(distance) else 0.0
                time_val = float(time_val) if time_val is not None and not pd.isna(time_val) else 0.0
                
                distance = distance / 1000  # Convert to kilometers
                time_val = time_val / 60  # Convert to minutes
                
                # Round distance and time values
                distance = round(distance, 2)
                time_val = round(time_val, 1)
                
            except (ValueError, TypeError) as e:
                print(f"Error processing distance/time values: {e}")
                distance = 0.1  # Default to 100 meters
                time_val = 10.0  # Default to 10 minutes
            
            results.append({
                'from_store': str(from_store),
                'to_store': str(to_store),
                'distance_km': distance,
                'time_minutes': time_val
            })
    
    return results
