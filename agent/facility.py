import requests
import sqlite3

def find_closest_facility(user_coords, db_path):
    """
    Finds the closest recycling facility to the user's coordinates using SQLite database and OSRM API.

    Args:
        user_coords (Tuple[float, float]): A tuple of the user's latitude and longitude.
        db_path (str): Path to the SQLite3 database file containing facility data.

    Returns:
        dict: Dictionary containing the closest facility's details or an error message if no facility is found.
    """
    osrm_url_template = "http://router.project-osrm.org/route/v1/driving/{from_lon},{from_lat};{to_lon},{to_lat}"
    
    # Connect to SQLite Database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, street, zip, district, lat, long, opening_time FROM recyclinghof")
        facilities = cursor.fetchall()

        closest_facility = None
        min_distance = float('inf')

        # Iterate over facilities in database to calculate distance
        for facility in facilities:
            name, street, zip_code, district, lat, lon, opening_time = facility
            osrm_url = osrm_url_template.format(
                from_lon=user_coords[1], from_lat=user_coords[0],
                to_lon=lon, to_lat=lat
            )
            
            # Extract distance in km and driving duration in min
            try:
                response = requests.get(osrm_url, params={"overview": "false"})
                response.raise_for_status()
                route_data = response.json()

                if route_data.get("routes"):
                    route = route_data["routes"][0]
                    distance_km = route["distance"] / 1000  
                    duration_min = route["duration"] / 60  

                    # Determine closest facility
                    if distance_km < min_distance:
                        min_distance = distance_km
                        closest_facility = {
                            "name": name,
                            "address": f"{street}, {zip_code} Frankfurt am Main",
                            "district": district,
                            "distance_km": distance_km,
                            "duration_min": duration_min,
                            "opening_time": opening_time,
                        }

            except requests.exceptions.RequestException:
                continue

        conn.close()

        return closest_facility or {"error": "No facilities found."}

    except Exception:
        return {"error": "Database query failed."}