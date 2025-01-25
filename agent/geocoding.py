import requests

def geocode_address(address: str):
    """
    Geocodes a user's address using the Nominatim API.

    Args:
        address (str): User's address as a string.

    Returns:
        Tuple[float, float]: A tuple containing the latitude and longitude of the address. Returns (None, None) if geocoding fails.
    """
    # Define API URL and parameters
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': 'WasteSeparationChatbot/1.0 (nghhoang@gmail.com)'}
    params = {'q': address, 'format': 'json', 'limit': 1}

    # Make API request
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except requests.exceptions.RequestException:
        pass

    return None, None
