import sqlite3
import requests

# Function to Geocode the input user address
def geocode_address(address):
    """
    Geocode a user's address using the Nominatim API.

    :param address: User's address as a string.
    :return: Tuple of (latitude, longitude) or (None, None) if geocoding fails.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'
    }
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except requests.exceptions.RequestException as e:
        print(f"Error geocoding address '{address}': {e}")
        
    return None, None

# Function to calculate the route, distance and duration between user address and facilities using OSRM
def calculate_route(from_coords, to_coords):
    """
    Calculate the route, distance, and duration between two coordinates using OSRM.

    :param from_coords: Tuple (latitude, longitude) for the starting point.
    :param to_coords: Tuple (latitude, longitude) for the destination.
    :return: Distance in kilometers, duration in minutes, or (None, None) if routing fails.
    """
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{from_coords[1]},{from_coords[0]};{to_coords[1]},{to_coords[0]}"
    
    try:
        response = requests.get(osrm_url, params={"overview": "false"})
        response.raise_for_status()
        route_data = response.json()
        if route_data.get("routes"):
            route = route_data["routes"][0]
            distance_km = route["distance"] / 1000  
            duration_min = route["duration"] / 60  
            return distance_km, duration_min
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during route calculation: {e}")
        
    return None, None

# Function to find the closest facility using the SQLite3 database
def find_closest_facility(user_coords, db_path):
    """
    Find the closest recycling facility to a user location.

    :param user_coords: Tuple (latitude, longitude) for the user's geolocation.
    :param db_path: Path to the SQLite3 database file.
    :return: Closest facility record with distance, duration, and opening time.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all facilities along with their geolocations and opening times from the database
    cursor.execute("SELECT name, street, zip, district, lat, long, opening_time FROM recyclinghof")
    facilities = cursor.fetchall()

    closest_facility = None
    min_distance = float('inf')

    for facility in facilities:
        (
            facility_name,
            street,
            zip_code,
            district,
            facility_lat,
            facility_long,
            opening_time,
        ) = facility
        to_coords = (facility_lat, facility_long)

        # Calculate the route distance and duration
        distance, duration = calculate_route(user_coords, to_coords)
        if distance is not None and distance < min_distance:
            min_distance = distance
            closest_facility = {
                "name": facility_name,
                "address": f"{street}, {zip_code} Frankfurt am Main",
                "district": district,
                "distance": distance,
                "duration": duration,
                "opening_time": opening_time,
            }

    conn.close()
    return closest_facility

# Example usage
if __name__ == "__main__":
    db_path = "database/waste_separation_frankfurt.db"
    user_address = "Bergerstraße 148, 60385 Frankfurt am Main"

    # Geocode the user's address
    user_coords = geocode_address(user_address)
    if user_coords == (None, None):
        print("Invalid user address. Unable to geocode.")
    else:
        # Find the closest facility
        closest_facility = find_closest_facility(user_coords, db_path)
        if closest_facility:
            print(f"The closest facility is '{closest_facility['name']}' at {closest_facility['address']} "
                  f"in the district '{closest_facility['district']}'.")
            print(f"Distance: {closest_facility['distance']:.2f} km")
            print(f"Travel time: {closest_facility['duration']:.2f} minutes")
            print(f"Opening hours: {closest_facility['opening_time']}")
        else:
            print("No facilities found.")
