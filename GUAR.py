# Google GSuite User Activity Report
# Updated by Shane Shook (c) 2025
# Note: requires ipstack key for geolocation; provide days, user_email, and ipstack keys in script

import datetime
import socket
import csv
import requests
import logging
import argparse
import time
import smtplib
from email.message import EmailMessage
from functools import lru_cache
from collections import Counter, defaultdict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ==========================
# CONFIGURABLE VARIABLES
# ==========================
DEFAULT_APPLICATIONS = 'all'  # Use 'all' or a comma-separated list of applications like ''login, drive, admin, calendar, token, groups, access_transparency, gmail, meet, etc.'
DEFAULT_EVENT_FILTERS = []  # Default set, ref specific filters at https://developers.google.com/admin-sdk/reports/reference/rest/v1/activities/list?utm_source=chatgpt.com
DEFAULT_OUTPUT = 'user_activity_report.csv'
SUMMARY_OUTPUT = 'user_activity_summary.csv'
ERROR_LOG = 'user_activity_errors.log'
DEFAULT_DAYS = 180  # Set to desired audit period from date the report is run
DEFAULT_IPSTACK_KEY = 'your_ipstack_access_key_here'  # Ref https://ipstack.com/api-key
DEFAULT_USER_EMAIL = 'all'  # Use 'all' or a comma-separated list of users like 'user1@domain.com, user2@domain.com...'
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 10

# Optional Email Config (set to None to disable)
EMAIL_REPORT = False
EMAIL_FROM = 'sender@domain.com'
EMAIL_TO = 'recipient@domain.com'
SMTP_SERVER = 'smtp.domain.com'
SMTP_PORT = 587
SMTP_USERNAME = 'sender@domain.com'
SMTP_PASSWORD = 'yourpassword'

# ==========================
# LOGGING SETUP
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler(ERROR_LOG)
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

# ==========================
# UTILITY FUNCTIONS
# ==========================
@lru_cache(maxsize=1024)
def get_hostname_from_ip(ip_address):
    try:
        return socket.gethostbyaddr(ip_address)[0]
    except Exception as e:
        error_logger.error(f"Hostname lookup failed for IP {ip_address}: {e}")
        return "Unknown"

@lru_cache(maxsize=1024)
def get_geolocation_from_ip(ip_address, access_key):
    try:
        url = f"http://api.ipstack.com/{ip_address}?access_key={access_key}"
        response = requests.get(url)
        data = response.json()
        return (
            data.get('country_name', 'Unknown'),
            data.get('city', 'Unknown'),
            data.get('latitude', 0.0),
            data.get('longitude', 0.0)
        )
    except Exception as e:
        error_logger.error(f"Geolocation lookup failed for IP {ip_address}: {e}")
        return "Unknown", "Unknown", 0.0, 0.0

# ==========================
# GOOGLE API AUTHENTICATION
# ==========================
def authenticate_with_reports(service_account_file, delegated_admin):
    scopes = [
        'https://www.googleapis.com/auth/admin.reports.audit.readonly',
        'https://www.googleapis.com/auth/admin.directory.user.readonly'
    ]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=scopes)
    delegated_credentials = credentials.with_subject(delegated_admin)
    reports_service = build('admin', 'reports_v1', credentials=delegated_credentials)
    directory_service = build('admin', 'directory_v1', credentials=delegated_credentials)
    return reports_service, directory_service

# ==========================
# FETCH USERS IN DOMAIN
# ==========================
def get_all_users(directory_service):
    users = []
    request = directory_service.users().list(customer='my_customer', maxResults=500, orderBy='email')
    while request is not None:
        try:
            response = request.execute()
            users.extend([user['primaryEmail'] for user in response.get('users', [])])
            request = directory_service.users().list_next(request, response)
        except HttpError as e:
            error_logger.error(f"Error fetching users: {e}")
            break
    return users

# ==========================
# RETRY WRAPPER FOR API CALLS
# ==========================
def execute_with_retry(request, user_email, app):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return request.execute()
        except HttpError as e:
            if e.resp.status in [403, 429, 500, 503]:
                retries += 1
                error_logger.error(f"Rate limit or server error for user {user_email}, app {app}. Retrying {retries}/{MAX_RETRIES}...")
                time.sleep(RETRY_WAIT_SECONDS * retries)
            else:
                raise
    raise Exception(f"Max retries exceeded for user {user_email}, app {app}")

# ==========================
# FETCH USER ACTIVITIES
# ==========================
def fetch_user_activities(service, user_email, days, application_names, event_filters, ipstack_key):
    if application_names == 'all':
        application_names = ['login', 'drive', 'admin', 'calendar', 'token', 'groups', 'access_transparency', 'gmail']
    else:
        application_names = application_names.split(',')
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days)
    start_rfc3339 = start_date.isoformat() + 'T00:00:00Z'
    end_rfc3339 = today.isoformat() + 'T23:59:59Z'

    all_activities = []
    summary_stats = defaultdict(Counter)

    for app in application_names:
        request = service.activities().list(
            userKey=user_email,
            applicationName=app,
            startTime=start_rfc3339,
            endTime=end_rfc3339
        )

        while request is not None:
            try:
                response = execute_with_retry(request, user_email, app)
                items = response.get('items', [])

                for activity in items:
                    events = activity.get('events', [])
                    for event in events:
                        if not event_filters or event.get('name') in event_filters:
                            ip_address = activity.get('ipAddress', 'Unknown')
                            hostname = get_hostname_from_ip(ip_address) if ip_address != 'Unknown' else 'Unknown'
                            country, city, latitude, longitude = get_geolocation_from_ip(ip_address, ipstack_key) if ip_address != 'Unknown' else ('Unknown', 'Unknown', 0.0, 0.0)

                            activity_record = {
                                'time': activity.get('id', {}).get('time', 'Unknown'),
                                'applicationName': app,
                                'eventName': event.get('name', 'Unknown'),
                                'userEmail': activity.get('actor', {}).get('email', user_email),
                                'ipAddress': ip_address,
                                'hostname': hostname,
                                'country': country,
                                'city': city,
                                'latitude': latitude,
                                'longitude': longitude,
                                'tenantIdentifier': 'YourTenantIdentifier',
                                'timeZone': 'YourTimeZone'
                            }

                            all_activities.append(activity_record)
                            summary_stats[app][event.get('name', 'Unknown')] += 1

                request = service.activities().list_next(request, response)

            except HttpError as e:
                error_logger.error(f"Error fetching activities for user {user_email}, app {app}: {e}")
                break

    return all_activities, summary_stats

# ==========================
# SEND EMAIL REPORT (OPTIONAL)
# ==========================
def send_email_report():
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Google Workspace Activity Report'
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg.set_content('Please find attached the user activity report and summary.')

        for file in [DEFAULT_OUTPUT, SUMMARY_OUTPUT, ERROR_LOG]:
            with open(file, 'rb') as f:
                msg.add_attachment(f.read(), maintype='text', subtype='csv', filename=file)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info("Email report sent successfully.")
    except Exception as e:
        error_logger.error(f"Failed to send email report: {e}")

# ==========================
# MAIN EXECUTION FUNCTION
# ==========================
def main():
    parser = argparse.ArgumentParser(description='Google Workspace User Activity Report')
    parser.add_argument('--service_account_file', required=True)
    parser.add_argument('--delegated_admin', required=True)
    parser.add_argument('--user_email', default=DEFAULT_USER_EMAIL)
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS)
    parser.add_argument('--ipstack_key', default=DEFAULT_IPSTACK_KEY)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--summary_output', default=SUMMARY_OUTPUT)
    parser.add_argument('--applications', default=DEFAULT_APPLICATIONS, help="Use 'all' or comma-separated application names like 'login,drive,gmail'")
    parser.add_argument('--event_filters', nargs='*', default=DEFAULT_EVENT_FILTERS)
    args = parser.parse_args()

    reports_service, directory_service = authenticate_with_reports(args.service_account_file, args.delegated_admin)

    if args.user_email.lower() == 'all':
        user_emails = get_all_users(directory_service)
    else:
        user_emails = [email.strip() for email in args.user_email.split(',')]

    all_activities = []
    all_summary_stats = defaultdict(Counter)

    for user_email in user_emails:
        logging.info(f"Fetching activity for {user_email}...")
        activities, summary_stats = fetch_user_activities(
            reports_service,
            user_email,
            args.days,
            args.applications,
            args.event_filters,
            args.ipstack_key
        )
        all_activities.extend(activities)
        for app, counts in summary_stats.items():
            for event_name, count in counts.items():
                all_summary_stats[app][event_name] += count

    if all_activities:
        sorted_activities = sorted(all_activities, key=lambda x: x['time'])
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = sorted_activities[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for activity in sorted_activities:
                writer.writerow(activity)
        logging.info(f'Activities written to {args.output}')

        with open(args.summary_output, 'w', newline='') as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(['Application', 'EventName', 'Count'])
            for app, events in all_summary_stats.items():
                for event_name, count in events.items():
                    writer.writerow([app, event_name, count])
        logging.info(f'Summary report written to {args.summary_output}')
    else:
        logging.info('No activities found.')

    if EMAIL_REPORT:
        send_email_report()

if __name__ == "__main__":
    main()
