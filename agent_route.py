import sqlite3
import requests

# Function to geocode an address using Nominatim with a custom User-Agent
def geocode_address(address):
    """
    Geocode an address using Nominatim API with a custom User-Agent.

    :param address: Address string to geocode.
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
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None, None

# Function to calculate the route, distance, and duration using OSRM
def calculate_route(user_coords, facility_coords):
    """
    Calculate the route, distance, and duration between two coordinates using OSRM.

    :param user_coords: Tuple (latitude, longitude) for the user's location.
    :param facility_coords: Tuple (latitude, longitude) for the facility location.
    :return: Distance in kilometers, duration in minutes, or (None, None) if routing fails.
    """
    osrm_url = "http://router.project-osrm.org/route/v1/driving"
    params = {
        "coordinates": f"{user_coords[1]},{user_coords[0]};{facility_coords[1]},{facility_coords[0]}",
        "overview": "false",
        "steps": "false"
    }
    try:
        response = requests.get(f"{osrm_url}/{params['coordinates']}", params={"overview": "false"})
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            route = data["routes"][0]
            distance_km = route["distance"] / 1000  # Convert meters to kilometers
            duration_min = route["duration"] / 60  # Convert seconds to minutes
            return distance_km, duration_min
    except requests.exceptions.RequestException as e:
        print(f"Error fetching route: {e}")
    return None, None

# Function to find the closest facility using the SQLite3 database
def find_closest_facility(user_address, db_path):
    """
    Find the closest recycling facility to a user address.

    :param user_address: User's address as a string.
    :param db_path: Path to the SQLite3 database file.
    :return: Closest facility record with distance and duration.
    """
    user_coords = geocode_address(user_address)
    if user_coords == (None, None):
        return "Invalid address. Unable to geocode."

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all facilities from the database
    cursor.execute("SELECT name, street, zip, district FROM recyclinghof")
    facilities = cursor.fetchall()

    closest_facility = None
    min_distance = float('inf')

    for facility in facilities:
        facility_name, street, zip_code, district = facility
        facility_address = f"{street}, {zip_code} Frankfurt am Main"
        facility_coords = geocode_address(facility_address)

        if facility_coords == (None, None):
            continue

        distance, duration = calculate_route(user_coords, facility_coords)
        if distance is not None and distance < min_distance:
            min_distance = distance
            closest_facility = {
                "name": facility_name,
                "address": facility_address,
                "district": district,
                "distance": distance,
                "duration": duration
            }

    conn.close()
    return closest_facility

# Example usage
if __name__ == "__main__":
    db_path = "database/waste_separation_frankfurt.db"  
    user_address = "Bergerstraße 148, 60385 Frankfurt am Main"

    closest_facility = find_closest_facility(user_address, db_path)
    if closest_facility:
        print(f"The closest facility is '{closest_facility['name']}' at {closest_facility['address']} "
              f"in the district '{closest_facility['district']}'.")
        print(f"Distance: {closest_facility['distance']:.2f} km")
        print(f"Travel time: {closest_facility['duration']:.2f} minutes")
    else:
        print("No facilities found.")
