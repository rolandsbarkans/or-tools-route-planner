from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
import os
from datetime import datetime
import pandas as pd

from scheduling_algorithm import build_vrp_data, solve_vrp, export_routes_for_ui, get_store_distances_times
from load_demand import load_generated_store_data

app = Flask(__name__)

# In-memory store replacing routes_ui.json
routes_cache = {}

@app.route('/')
def index():
    return send_from_directory('.', 'interface.html')

@app.route('/api/stores', methods=['GET'])
def get_stores():
    try:
        selected_day = request.args.get('day')

        if not selected_day:
            return jsonify({
                'success': False,
                'error': 'Day parameter is required'
            }), 400

        data_dir = Path('data')

        stores_df = load_generated_store_data(selected_day, data_dir, data_dir / 'store_metadata.csv')
        
        stores = []
        for _, row in stores_df.iterrows():
            def safe_float(value, default=None):
                try:
                    return float(value) if pd.notna(value) else default
                except (ValueError, TypeError):
                    return default
            
            store_data = {
                'id': str(row.get('store_id', '')),
                'name': str(row.get('store_name', 'N/A')),
                'volume': safe_float(row.get('total_v', 0), 0),
                'latitude': safe_float(row.get('latitude')),
                'longitude': safe_float(row.get('longitude')),
                'address': str(row.get('address', ''))
            }
            
            if pd.isna(store_data['address']):
                store_data['address'] = ''
                
            stores.append(store_data)
            
        return jsonify({
            'success': True,
            'stores': stores
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/generate_routes', methods=['POST'])
def generate_routes():
    try:
        request_data = request.json
        
        selected_day = request_data.get('selected_day')
        if not selected_day:
            return jsonify({'error': 'Day parameter is required (e.g., monday)'}), 400
            
        num_vehicles = int(request_data.get('num_vehicles', 17))
        max_volume = float(request_data.get('max_volume', 200))
        max_stops = int(request_data.get('max_stops', 7))
        max_edge_distance = float(request_data.get('max_edge_distance', 30.0))
        
        data_dir = Path('data')
        
        available_stores = request_data.get('available_stores', [])
        
        if available_stores:
            stores_df = pd.DataFrame(available_stores)
            
            if 'store_address' in stores_df.columns:
                stores_df = stores_df.rename(columns={'store_address': 'address'})
            
            stores_df['store_id'] = stores_df['store_id'].astype(str)
            stores_df['total_v'] = pd.to_numeric(stores_df['total_v'], errors='coerce').fillna(0)
            stores_df['latitude'] = pd.to_numeric(stores_df['latitude'], errors='coerce')
            stores_df['longitude'] = pd.to_numeric(stores_df['longitude'], errors='coerce')
            
            stores_df = stores_df.dropna(subset=['latitude', 'longitude'])
            stores_df = stores_df[stores_df['total_v'] > 0]
            
        else:
            stores_df = load_generated_store_data(selected_day, data_dir, data_dir / 'store_metadata.csv')
        
        if stores_df.empty:
            return jsonify({'error': 'No store data available for route generation'}), 400
        
        vrp_data = build_vrp_data(
            store_data=stores_df,
            data_dir=data_dir,
            num_vehicles=num_vehicles,
            vehicle_capacity=max_volume,
            max_route_time_minutes=60000,
            allow_drop=True,
            drop_penalty=100000,
            unloading_time_minutes=20,
            max_edge_distance_m=max_edge_distance * 1000,
            max_stops_per_route=max_stops
        )
        
        num_stores = len(stores_df)
        if num_stores <= 10:
            time_limit = 3
        elif num_stores <= 30:
            time_limit = 10
        elif num_stores <= 100:
            time_limit = 15
        elif num_stores <= 150:
            time_limit = 20
        else:
            time_limit = 40
        
        solution = solve_vrp(vrp_data, time_limit_seconds=time_limit)
        
        if not solution or 'routes' not in solution or not solution['routes']:
            return jsonify({'error': 'No solution found'}), 400
            
        routes_data = export_routes_for_ui(
            data=vrp_data,
            result=solution,
            data_dir=data_dir,
            unloading_time_minutes=20
        )
        
        routes_data['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'selectedDay': selected_day,
                'numVehicles': num_vehicles,
                'maxVolume': max_volume,
                'maxStops': max_stops,
                'maxEdgeDistance': max_edge_distance
            }
        }
        
        routes_cache['data'] = routes_data
        
        return jsonify(routes_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_routes', methods=['GET'])
def get_routes():
    data = routes_cache.get('data')
    if not data:
        return jsonify({
            'routes': [],
            'dropped_stores': [],
            'constraints': {},
            'summary': {},
            'metadata': {}
        }), 200
    return jsonify(data)

@app.route('/clear_routes', methods=['POST'])
def clear_routes():
    routes_cache.clear()
    return jsonify({'success': True, 'message': 'Routes data cleared successfully'})

@app.route('/update_routes', methods=['POST'])
def update_routes():
    data = request.json
    routes_cache['data'] = data
    return jsonify({'success': True, 'message': 'Routes updated successfully'})

@app.route('/get_store_distances', methods=['POST'])
def get_store_distances():
    store_ids = request.json.get('store_ids', [])
    data_dir = Path('data')
    distances_times = get_store_distances_times(store_ids, data_dir)
    return jsonify({
        'success': True,
        'distances_times': distances_times
    })

if __name__ == '__main__':
    app.run(debug=True, port=1234)