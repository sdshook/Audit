# Google GSuite User Activity Report
# Created by Shane Shook (c) 2024
# Note: requires ipstack key for geolocation; provide days, user_email, and ipstack keys in script

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime
import socket
import csv
import requests

# Function to perform NSLookup for an IP address
def get_hostname_from_ip(ip_address):
    try:
        # Perform NSLookup (DNS reverse lookup)
        hostname = socket.gethostbyaddr(ip_address)[0]
        return hostname
    except:
        # If NSLookup fails, return "Unknown"
        return "Unknown"

# Function to perform geolocation lookup for an IP address using ipstack API
def get_geolocation_from_ip(ip_address, access_key):
    try:
        # ipstack API endpoint
        url = f"http://api.ipstack.com/{ip_address}?access_key={access_key}"

        # Make request to ipstack API
        response = requests.get(url)

        # Parse JSON response
        data = response.json()

        # Extract geolocation information
        country = data.get('country_name', 'Unknown')
        city = data.get('city', 'Unknown')
        latitude = data.get('latitude', 0.0)
        longitude = data.get('longitude', 0.0)

        return country, city, latitude, longitude
    except:
        # If geolocation lookup fails, return "Unknown" values
        return "Unknown", "Unknown", 0.0, 0.0

# Set up authentication with Google Vault API using service account credentials
def authenticate_with_vault():
    # Path to your service account key file
    service_account_file = 'path/to/service_account.json'

    # Scopes required for accessing the Google Vault API
    scopes = ['https://www.googleapis.com/auth/ediscovery']

    # Authenticate with Google Vault API using service account credentials
    credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=scopes)

    # Build the Google Vault API client
    vault_service = build('vault', 'v1', credentials=credentials)

    return vault_service

# Function to fetch activities from Google Vault
def fetch_vault_activities(vault_service, days, user_email):
    # Get today's date
    today = datetime.date.today()

    # Calculate start date based on the specified number of days
    start_date = today - datetime.timedelta(days=days)

    # Convert dates to RFC3339 format required by the API
    start_date_rfc3339 = start_date.isoformat() + "T00:00:00Z"
    end_date_rfc3339 = today.isoformat() + "T23:59:59Z"

    # Example: list matters
    matters = vault_service.matters().list().execute()

    # Iterate through matters and fetch activities
    activities = []
    for matter in matters:
        # Fetch activities for each matter
        matter_activities = vault_service.matters().activities().list(matterId=matter['matterId'], startDate=start_date_rfc3339, endDate=end_date_rfc3339, userKey=user_email).execute()

        # Add activities to the list
        activities.extend(matter_activities)

    return activities

# Main function
def main():
    # Authenticate with Google Vault API
    vault_service = authenticate_with_vault()

    # Define the number of days for the date filter
    days = 30

    # Define the user email to request information about
    user_email = 'user@example.com'

    # Define your ipstack API access key
    ipstack_access_key = 'your_ipstack_access_key'

    # Fetch activities from Google Vault for the specified user
    activities = fetch_vault_activities(vault_service, days, user_email)

    # Process activities
    processed_activities = []
    for activity in activities:
        # Process activity as needed
        # Example: Perform NSLookup and geolocation lookup for IP addresses
        ip_address = activity.get('ipAddress')
        if ip_address:
            hostname = get_hostname_from_ip(ip_address)
            country, city, latitude, longitude = get_geolocation_from_ip(ip_address, ipstack_access_key)
            activity['hostname'] = hostname
            activity['country'] = country
            activity['city'] = city
            activity['latitude'] = latitude
            activity['longitude'] = longitude

        # Add tenant identifier and time zone to each activity
        activity['tenantIdentifier'] = 'YourTenantIdentifier'
        activity['timeZone'] = 'YourTimeZone'

        # Add processed activity to the list
        processed_activities.append(activity)

    # Sort activities by date
    sorted_activities = sorted(processed_activities, key=lambda x: x['time'])

    # Define CSV file path
    csv_file_path = 'google_vault_activities.csv'

    # Write activities to CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = sorted_activities[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for activity in sorted_activities:
            writer.writerow(activity)

    print(f'Activities written to {csv_file_path}')

if __name__ == "__main__":
    main()
