#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export privileged admins from On-Prem AD and Azure AD (PIM-aware), plus
app/service principal credentials and their recent changes.

(c) 2025 Shane D. Shook, All Rights Reserved.

DESCRIPTION
  Outputs:
    - OnPrem_AD_Admins.csv                (expanded on-prem privileged groups + optional extra groups)
    - AzureAD_Admins_Active.csv           (PIM active assignments; users/SPs; group expansion)
    - AzureAD_Admins_Eligible.csv         (PIM eligible assignments; users/SPs; group expansion)
    - AzureAD_AppSecrets.csv              (flattened secrets/certs for SPs holding privileged roles)
    - AzureAD_AppSecretChanges.csv        (audit log events for secret/cert adds/removes/updates)
    - All_Admins.csv                      (merged view across on-prem + Azure)

PARAMETERS / CLI
  --mode OP|AZ|BOTH         (default: BOTH)
  --outdir PATH             (default: .)
  Azure auth options (choose one):
    * Device Code: --auth devicecode --tenant TENANT_ID --client-id CLIENT_ID
    * Client Secret: --auth clientsecret --tenant TENANT_ID --client-id CLIENT_ID --client-secret SECRET
  On-Prem AD:
    --ad-server SERVER (e.g., ldaps://dc1.example.com:636 or dc1.example.com)
    --ad-user USERNAME (e.g., 'example\\alice' or 'alice@example.com')
    --ad-pass PASSWORD
    --ad-base-dn "DC=example,DC=com" (optional; else auto from RootDSE)
    --ad-extra-groups "CN=Tier0 Admins, CN=Ops Admins"   (optional; adds to default list)
    --ad-tls-skip-verify                                  (optional; allow insecure LDAPS)

Time filters (mutually exclusive; default is --all):
  --all
  --start MMDDYYYY --end MMDDYYYY   (inclusive)
  --last N                          (last N days)

REQUIREMENTS
  pip install requests msal ldap3 python-dateutil

HOW TO RUN
  # Everything (device code auth), no time filter:
  python AdminCreds.py --mode BOTH --outdir C:\Exports --auth devicecode --tenant <TENANT_ID> --client-id <CLIENT_ID> --all

  # Azure-only, last 120 days of app secret changes (client secret auth):
  python AdminCreds.py --mode AZ --outdir C:\Exports --auth clientsecret --tenant <TENANT_ID> --client-id <CLIENT_ID> --client-secret <SECRET> --last 120

  # Both, explicit range (Jul 1–Aug 31, 2025), with on-prem LDAP creds:
  python AdminCreds.py --mode BOTH --outdir C:\Exports --auth devicecode --tenant <TENANT_ID> --client-id <CLIENT_ID> \
      --start 07012025 --end 08312025 \
      --ad-user 'example\\svc_reader' --ad-pass '***' --ad-server 'ldaps://dc1.example.com:636' --ad-base-dn "DC=example,DC=com"
"""

import argparse
import csv
import os
import sys
import time
import ssl
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
from email.utils import parsedate_to_datetime

import requests
from dateutil import parser as dtparser

# Optional import for on-prem AD; handle if missing
try:
    from ldap3 import Server, Connection, ALL, NTLM, SUBTREE, Tls, SIMPLE
    LDAP_AVAILABLE = True
except Exception:
    LDAP_AVAILABLE = False

# ----------------------
# Constants / Scopes
# ----------------------
GRAPH_AUTHORITY_FMT = "https://login.microsoftonline.com/{tenant}"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Delegated scopes for device-code
SCOPES_DELEGATED = [
    "Directory.Read.All",
    "RoleManagement.Read.Directory",
    "Group.Read.All",
    "Application.Read.All",
    "AuditLog.Read.All",
]
# App permissions for client-credential
SCOPES_APP = ["https://graph.microsoft.com/.default"]

NEEDED_PERMISSIONS = (
    "RoleManagement.Read.Directory, Directory.Read.All, Group.Read.All, "
    "Application.Read.All, AuditLog.Read.All"
)

PRIV_GROUPS_OP = [
    'Administrators','Domain Admins','Enterprise Admins','Schema Admins',
    'Account Operators','Backup Operators','Server Operators','Print Operators',
    'DnsAdmins','Group Policy Creator Owners','Key Admins','Enterprise Key Admins'
]

# UAC flags
ADS_UF_DONT_REQUIRE_PREAUTH = 0x0040000   # 262144
ADS_UF_USE_DES_KEY_ONLY     = 0x0200000   # 2097152
ADS_UF_ACCOUNTDISABLE       = 0x0000002
ADS_UF_DONT_EXPIRE_PASSWD   = 0x00010000

# ----------------------
# Globals
# ----------------------
_session = requests.Session()

# ----------------------
# Utility
# ----------------------
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_mmddyyyy(s: str) -> datetime:
    return datetime.strptime(s, "%m%d%Y").replace(tzinfo=timezone.utc)

def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def window_overlap(item_start: Optional[datetime], item_end: Optional[datetime],
                   win_start: Optional[datetime], win_end: Optional[datetime]) -> bool:
    # No filter
    if not win_start and not win_end:
        return True
    is_ = item_start or datetime.min.replace(tzinfo=timezone.utc)
    ie_ = item_end   or datetime.max.replace(tzinfo=timezone.utc)
    ws = win_start or datetime.min.replace(tzinfo=timezone.utc)
    we = win_end   or datetime.max.replace(tzinfo=timezone.utc)
    return is_ <= we and ie_ >= ws

# ----------------------
# CSV helpers (always write headers with BOM)
# ----------------------
def write_csv_with_header(path: str, rows: List[Dict[str, Any]], header: List[str]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        if rows:
            for r in rows:
                for h in header:
                    r.setdefault(h, None)
            w.writerows(rows)
    log(f"Wrote: {path} ({len(rows)} rows)")

# ----------------------
# MSAL Auth
# ----------------------
def acquire_token_devicecode(tenant: str, client_id: str) -> str:
    try:
        import msal
    except ImportError:
        log("ERROR: msal not installed. pip install msal")
        sys.exit(1)
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=GRAPH_AUTHORITY_FMT.format(tenant=tenant),
    )
    flow = app.initiate_device_flow(scopes=SCOPES_DELEGATED)
    if "user_code" not in flow:
        raise RuntimeError("Failed to create device flow")
    print(flow["message"])
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(f"Token error: {result}")
    return result["access_token"]

def acquire_token_clientsecret(tenant: str, client_id: str, client_secret: str) -> str:
    try:
        import msal
    except ImportError:
        log("ERROR: msal not installed. pip install msal")
        sys.exit(1)
    app = msal.ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=GRAPH_AUTHORITY_FMT.format(tenant=tenant),
    )
    result = app.acquire_token_for_client(scopes=SCOPES_APP)
    if "access_token" not in result:
        raise RuntimeError(f"Token error: {result}")
    return result["access_token"]

# ----------------------
# Graph helpers with retry + Session reuse
# ----------------------
def graph_get(url: str, token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    backoff = 1.0
    for attempt in range(6):
        r = _session.get(url, headers=headers, params=params, timeout=60)
        if r.status_code in (429, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    sleep_for = float(ra)  # seconds
                except ValueError:
                    try:
                        dt = parsedate_to_datetime(ra)
                        sleep_for = max(0.0, (dt - datetime.now(dt.tzinfo)).total_seconds())
                    except Exception:
                        sleep_for = backoff
            else:
                sleep_for = backoff
            time.sleep(min(sleep_for, 30))
            backoff = min(backoff * 2, 30)
            continue
        if r.status_code == 403:
            raise RuntimeError(f"Graph GET {url} failed 403 (Forbidden). "
                               f"Verify app consent/permissions: {NEEDED_PERMISSIONS}. Response: {r.text}")
        if not r.ok:
            raise RuntimeError(f"Graph GET {url} failed {r.status_code}: {r.text}")
        return r.json()
    raise RuntimeError("Graph GET failed after retries")

def graph_list_all(url: str, token: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    next_url = url
    next_params = params or {}
    while next_url:
        data = graph_get(next_url, token, next_params)
        items.extend(data.get("value", []))
        next_url = data.get("@odata.nextLink")
        next_params = None  # nextLink already includes params
    return items

# Caches
_user_cache: Dict[str, Dict[str, Any]] = {}
_sp_cache: Dict[str, Dict[str, Any]] = {}
_group_cache: Dict[str, Dict[str, Any]] = {}
_app_cache: Dict[str, Dict[str, Any]] = {}
_roledef_cache: Dict[str, Dict[str, Any]] = {}

def get_role_definitions(token: str) -> Dict[str, Dict[str, Any]]:
    global _roledef_cache
    if _roledef_cache:
        return _roledef_cache
    url = f"{GRAPH_BASE}/roleManagement/directory/roleDefinitions"
    for rd in graph_list_all(url, token):
        _roledef_cache[rd["id"]] = rd
    return _roledef_cache

def get_user(token: str, user_id: str) -> Optional[Dict[str, Any]]:
    if user_id in _user_cache:
        return _user_cache[user_id]
    try:
        data = graph_get(f"{GRAPH_BASE}/users/{user_id}?$select=id,displayName,userPrincipalName,accountEnabled", token)
    except Exception:
        data = None
    _user_cache[user_id] = data
    return data

def get_sp(token: str, sp_id: str) -> Optional[Dict[str, Any]]:
    if sp_id in _sp_cache:
        return _sp_cache[sp_id]
    try:
        data = graph_get(f"{GRAPH_BASE}/servicePrincipals/{sp_id}?$select=id,displayName,accountEnabled,appId", token)
    except Exception:
        data = None
    _sp_cache[sp_id] = data
    return data

def get_group(token: str, group_id: str) -> Optional[Dict[str, Any]]:
    if group_id in _group_cache:
        return _group_cache[group_id]
    try:
        data = graph_get(f"{GRAPH_BASE}/groups/{group_id}?$select=id,displayName", token)
    except Exception:
        data = None
    _group_cache[group_id] = data
    return data

def get_application_by_appid(token: str, app_id: str) -> Optional[Dict[str, Any]]:
    if not app_id:
        return None
    if app_id in _app_cache:
        return _app_cache[app_id]
    items = graph_list_all(
        f"{GRAPH_BASE}/applications",
        token,
        params={"$filter": f"appId eq '{app_id}'", "$select": "appId,displayName,passwordCredentials,keyCredentials"}
    )
    data = items[0] if items else None
    _app_cache[app_id] = data
    return data

def expand_group_transitive_members(token: str, group_id: str) -> List[Dict[str, Any]]:
    url = f"{GRAPH_BASE}/groups/{group_id}/transitiveMembers?$select=id,displayName,userPrincipalName,@odata.type"
    return graph_list_all(url, token)

# ----------------------
# Azure collectors
# ----------------------
def _row_template(
    Source="AzureAD", Scope="", RoleOrGroup="", ObjectType="", DisplayName=None, UserPrincipalName=None,
    AccountEnabled=None, ObjectId=None, InheritedFromGroup=None, SamAccountName=None, Enabled=None,
    LastLogonDate=None, PasswordLastSet=None, DoesNotRequirePreAuth=None, UseDESKeyOnly=None, PasswordNeverExpires=None,
    DirectoryScopeId=None, AssignmentStart=None, AssignmentEnd=None
):
    return dict(
        Source=Source, Scope=Scope, RoleOrGroup=RoleOrGroup, ObjectType=ObjectType, DisplayName=DisplayName,
        UserPrincipalName=UserPrincipalName, AccountEnabled=AccountEnabled, ObjectId=ObjectId,
        InheritedFromGroup=InheritedFromGroup, SamAccountName=SamAccountName, Enabled=Enabled,
        LastLogonDate=LastLogonDate, PasswordLastSet=PasswordLastSet, DoesNotRequirePreAuth=DoesNotRequirePreAuth,
        UseDESKeyOnly=UseDESKeyOnly, PasswordNeverExpires=PasswordNeverExpires,
        DirectoryScopeId=DirectoryScopeId, AssignmentStart=AssignmentStart, AssignmentEnd=AssignmentEnd
    )

def collect_azure_roles(token: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (active_rows, eligible_rows) flattened with group expansion."""
    role_defs = get_role_definitions(token)

    def to_row_for_principal(scope_label: str, role_name: str, principal: Dict[str, Any],
                             inherited_from_group: Optional[str] = None) -> List[Dict[str, Any]]:
        typ = (principal.get("principalType")
               or principal.get("principal", {}).get("@odata.type", "").replace("#microsoft.graph.", "")
               or principal.get("@odata.type", "").replace("#microsoft.graph.", ""))

        principal_id = principal.get("principalId") or principal.get("id")
        scope_id = principal.get("directoryScopeId")
        start_dt = principal.get("startDateTime")
        end_dt   = principal.get("endDateTime")

        rows: List[Dict[str, Any]] = []
        if typ == "user":
            u = get_user(token, principal_id)
            rows.append(_row_template(
                Scope=scope_label, RoleOrGroup=role_name, ObjectType="user",
                DisplayName=(u or {}).get("displayName"), UserPrincipalName=(u or {}).get("userPrincipalName"),
                AccountEnabled=(u or {}).get("accountEnabled"), ObjectId=(u or {}).get("id"),
                InheritedFromGroup=inherited_from_group,
                DirectoryScopeId=scope_id, AssignmentStart=start_dt, AssignmentEnd=end_dt
            ))
            return rows
        if typ == "servicePrincipal":
            sp = get_sp(token, principal_id)
            rows.append(_row_template(
                Scope=scope_label, RoleOrGroup=role_name, ObjectType="servicePrincipal",
                DisplayName=(sp or {}).get("displayName"), UserPrincipalName=None,
                AccountEnabled=(sp or {}).get("accountEnabled"), ObjectId=(sp or {}).get("id"),
                InheritedFromGroup=inherited_from_group,
                DirectoryScopeId=scope_id, AssignmentStart=start_dt, AssignmentEnd=end_dt
            ))
            return rows
        if typ == "group":
            g = get_group(token, principal_id)
            gname = (g or {}).get("displayName")
            members = expand_group_transitive_members(token, principal_id)
            for m in members:
                mtype = m.get("@odata.type", "").replace("#microsoft.graph.", "")
                if mtype == "user":
                    u = get_user(token, m.get("id"))
                    rows.append(_row_template(
                        Scope=scope_label, RoleOrGroup=role_name, ObjectType="user",
                        DisplayName=(u or {}).get("displayName"), UserPrincipalName=(u or {}).get("userPrincipalName"),
                        AccountEnabled=(u or {}).get("accountEnabled"), ObjectId=(u or {}).get("id"),
                        InheritedFromGroup=gname,
                        DirectoryScopeId=scope_id, AssignmentStart=start_dt, AssignmentEnd=end_dt
                    ))
                elif mtype == "servicePrincipal":
                    sp = get_sp(token, m.get("id"))
                    rows.append(_row_template(
                        Scope=scope_label, RoleOrGroup=role_name, ObjectType="servicePrincipal",
                        DisplayName=(sp or {}).get("displayName"), UserPrincipalName=None,
                        AccountEnabled=(sp or {}).get("accountEnabled"), ObjectId=(sp or {}).get("id"),
                        InheritedFromGroup=gname,
                        DirectoryScopeId=scope_id, AssignmentStart=start_dt, AssignmentEnd=end_dt
                    ))
            return rows

        # default catch-all
        rows.append(_row_template(
            Scope=scope_label, RoleOrGroup=role_name, ObjectType=typ or "unknown",
            DisplayName=None, UserPrincipalName=None, AccountEnabled=None, ObjectId=principal_id,
            InheritedFromGroup=inherited_from_group,
            DirectoryScopeId=scope_id, AssignmentStart=start_dt, AssignmentEnd=end_dt
        ))
        return rows

    active_rows: List[Dict[str, Any]] = []
    elig_rows: List[Dict[str, Any]] = []

    # Active assignments
    act_items = graph_list_all(f"{GRAPH_BASE}/roleManagement/directory/roleAssignmentScheduleInstances", token)
    for it in act_items:
        role_name = (role_defs.get(it.get("roleDefinitionId")) or {}).get("displayName") or it.get("roleDefinitionId")
        active_rows.extend(to_row_for_principal("Active", role_name, it))

    # Eligible assignments
    el_items = graph_list_all(f"{GRAPH_BASE}/roleManagement/directory/roleEligibilityScheduleInstances", token)
    for it in el_items:
        role_name = (role_defs.get(it.get("roleDefinitionId")) or {}).get("displayName") or it.get("roleDefinitionId")
        elig_rows.extend(to_row_for_principal("Eligible", role_name, it))

    # Sort for stable output
    def sorter(r): return (r["RoleOrGroup"] or "", r["ObjectType"] or "", r["UserPrincipalName"] or "", r["DisplayName"] or "")
    active_rows.sort(key=sorter)
    elig_rows.sort(key=sorter)
    return active_rows, elig_rows

def collect_app_secrets(token: str, active_rows: List[Dict[str, Any]], elig_rows: List[Dict[str, Any]],
                        win_start: Optional[datetime], win_end: Optional[datetime]) -> List[Dict[str, Any]]:
    """For SPs that hold privileged roles, dump PasswordCredentials and KeyCredentials."""
    sp_ids = sorted({r["ObjectId"] for r in active_rows + elig_rows if r.get("ObjectType") == "servicePrincipal" and r.get("ObjectId")})
    out: List[Dict[str, Any]] = []
    for sp_id in sp_ids:
        sp = get_sp(token, sp_id)
        if not sp:
            continue
        app = get_application_by_appid(token, sp.get("appId"))
        if not app:
            # SP without corresponding application (e.g., gallery/first-party) — skip secrets
            continue
        app_name = app.get("displayName")
        app_id = app.get("appId")
        sp_name = sp.get("displayName")
        # Secrets
        for p in app.get("passwordCredentials", []) or []:
            try:
                sdt = dtparser.isoparse(p.get("startDateTime")) if p.get("startDateTime") else None
                edt = dtparser.isoparse(p.get("endDateTime")) if p.get("endDateTime") else None
            except Exception:
                sdt = edt = None
            if win_start or win_end:
                if not window_overlap(sdt, edt, win_start, win_end):
                    continue
            out.append(dict(
                AppId=app_id,
                ApplicationDisplayName=app_name,
                ServicePrincipalId=sp_id,
                ServicePrincipalDisplayName=sp_name,
                CredentialType="Secret",
                CredentialDisplayName=p.get("displayName"),
                StartDateTime=p.get("startDateTime"),
                EndDateTime=p.get("endDateTime"),
                KeyId=p.get("keyId"),
                Hint=None
            ))
        # Certificates
        for k in app.get("keyCredentials", []) or []:
            try:
                sdt = dtparser.isoparse(k.get("startDateTime")) if k.get("startDateTime") else None
                edt = dtparser.isoparse(k.get("endDateTime")) if k.get("endDateTime") else None
            except Exception:
                sdt = edt = None
            if win_start or win_end:
                if not window_overlap(sdt, edt, win_start, win_end):
                    continue
            hint = None
            cki = k.get("customKeyIdentifier")
            if cki is not None:
                if isinstance(cki, str):
                    hint = cki
                else:
                    try:
                        import base64
                        hint = base64.b64encode(cki).decode("ascii")
                    except Exception:
                        hint = None
            out.append(dict(
                AppId=app_id,
                ApplicationDisplayName=app_name,
                ServicePrincipalId=sp_id,
                ServicePrincipalDisplayName=sp_name,
                CredentialType="Certificate",
                CredentialDisplayName=k.get("displayName"),
                StartDateTime=k.get("startDateTime"),
                EndDateTime=k.get("endDateTime"),
                KeyId=k.get("keyId"),
                Hint=hint
            ))
    # Sort
    out.sort(key=lambda r: (r.get("ApplicationDisplayName") or "", r.get("CredentialType") or "", r.get("EndDateTime") or ""))
    return out

def collect_app_secret_changes(token: str, win_start: Optional[datetime], win_end: Optional[datetime]) -> List[Dict[str, Any]]:
    """Directory audit events related to app/SP credential adds/removes/updates within window."""
    params = None
    if win_start and win_end:
        params = {"$filter": f"activityDateTime ge {_to_utc_iso(win_start)} and activityDateTime le {_to_utc_iso(win_end)}"}
    elif win_start:
        params = {"$filter": f"activityDateTime ge {_to_utc_iso(win_start)}"}
    elif win_end:
        params = {"$filter": f"activityDateTime le {_to_utc_iso(win_end)}"}

    items = graph_list_all(f"{GRAPH_BASE}/auditLogs/directoryAudits", token, params=params)
    wanted = []
    for e in items:
        ad = (e.get("activityDisplayName") or "").lower()
        if (
            ("serviceprincipal" in ad and any(x in ad for x in ["password","key","certificate","credential"])) or
            ("application" in ad and any(x in ad for x in ["password","key","certificate","credential"])) or
            ("certificates and secrets" in ad)
        ):
            user_info = (e.get("initiatedBy") or {}).get("user") or {}
            initiated_by = user_info.get("userPrincipalName") or user_info.get("displayName")
            target = (e.get("targetResources") or [None])[0] or {}
            wanted.append(dict(
                ActivityDateTime=e.get("activityDateTime"),
                ActivityDisplayName=e.get("activityDisplayName"),
                InitiatedBy=initiated_by,
                Category=e.get("category"),
                TargetType=target.get("type"),
                TargetId=target.get("id"),
                AdditionalDetails=json.dumps(e.get("additionalDetails"), ensure_ascii=False, separators=(",", ":"), default=str)
            ))
    # Defensive client-side time filter
    if win_start or win_end:
        def in_range(s: Optional[str]) -> bool:
            if not s:
                return False
            try:
                t = dtparser.isoparse(s)
            except Exception:
                return False
            if win_start and t < win_start:
                return False
            if win_end and t > win_end:
                return False
            return True
        wanted = [r for r in wanted if in_range(r["ActivityDateTime"])]
    wanted.sort(key=lambda r: r.get("ActivityDateTime") or "")
    return wanted

# ----------------------
# On-Prem AD via ldap3
# ----------------------
def collect_onprem_admins(ad_server: Optional[str], ad_user: Optional[str], ad_pass: Optional[str],
                          ad_base_dn: Optional[str], ad_tls_skip_verify: bool, ad_extra_groups: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not LDAP_AVAILABLE:
        log("ldap3 not installed; skipping on-prem AD. pip install ldap3")
        return []

    if not (ad_server and ad_user and ad_pass):
        log("Missing --ad-server/--ad-user/--ad-pass; skipping on-prem AD.")
        return []

    # Connection setup with safe host:port parsing
    use_ssl = ad_server.lower().endswith(":636") or ad_server.lower().startswith("ldaps://")
    raw = ad_server.replace("ldaps://", "").replace("ldap://", "")
    if ":" in raw:
        host, port_str = raw.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            host, port = raw, (636 if use_ssl else 389)
    else:
        host, port = raw, (636 if use_ssl else 389)

    if use_ssl:
        tls = Tls(validate=ssl.CERT_NONE) if ad_tls_skip_verify else Tls(validate=ssl.CERT_REQUIRED)
    else:
        tls = None

    server = Server(host, port=port, use_ssl=use_ssl, get_info=ALL, tls=tls, connect_timeout=10)

    if "\\" in ad_user:
        conn = Connection(server, user=ad_user, password=ad_pass, authentication=NTLM, auto_bind=True)
    else:
        conn = Connection(server, user=ad_user, password=ad_pass, authentication=SIMPLE, auto_bind=True)

    # Try to auto-discover base DN if missing
    if not ad_base_dn:
        conn.search(search_base="", search_filter="(objectClass=*)", search_scope="BASE", attributes=["defaultNamingContext"])
        if conn.entries:
            ad_base_dn = str(conn.entries[0]["defaultNamingContext"])
        else:
            raise RuntimeError("Could not determine base DN; please pass --ad-base-dn.")

    rows: List[Dict[str, Any]] = []

    # Helper: expand group membership transitively to users
    def expand_group_members(group_dn: str) -> List[str]:
        users: List[str] = []
        to_visit = [group_dn]
        seen = set()
        while to_visit:
            dn = to_visit.pop()
            if dn in seen:
                continue
            seen.add(dn)
            conn.search(search_base=dn, search_filter="(objectClass=group)", search_scope="BASE", attributes=["member","objectClass"])
            if not conn.entries:
                continue
            entry = conn.entries[0]
            members = entry["member"].values if "member" in entry else []
            for m in members:
                conn.search(search_base=m, search_filter="(objectClass=*)", search_scope="BASE", attributes=["objectClass"])
                if not conn.entries:
                    continue
                oc = [str(x).lower() for x in conn.entries[0]["objectClass"].values]
                if "group" in oc:
                    to_visit.append(m)  # nested group
                elif "user" in oc or "person" in oc:
                    users.append(m)
        return users

    # Resolve group CN -> DN
    def cn_to_dn(cn: str) -> Optional[str]:
        conn.search(search_base=ad_base_dn, search_filter=f"(&(objectClass=group)(cn={cn}))", search_scope=SUBTREE, attributes=["distinguishedName"])
        if conn.entries:
            return str(conn.entries[0]["distinguishedName"])
        return None

    # Convert AD large integer times
    def ad_time_to_dt(v):
        try:
            v = int(v)
            if v == 0:
                return None
            epoch_start = datetime(1601, 1, 1, tzinfo=timezone.utc)
            return epoch_start + timedelta(microseconds=v/10)
        except Exception:
            return None

    # Build group set
    groups = PRIV_GROUPS_OP[:]
    if ad_extra_groups:
        groups += [g for g in ad_extra_groups if g]

    # For each privileged group, expand and fetch selected attributes
    for g in groups:
        dn = cn_to_dn(g) if not g.upper().startswith("CN=") else g  # if full DN passed, use directly
        if not dn:
            log(f"Warn: group '{g}' not found in AD (CN search).")
            continue
        user_dns = expand_group_members(dn)
        for udn in user_dns:
            conn.search(
                search_base=udn, search_filter="(objectClass=user)", search_scope="BASE",
                attributes=["sAMAccountName","userAccountControl","lastLogonTimestamp","pwdLastSet","userPrincipalName","displayName"]
            )
            if not conn.entries:
                continue
            e = conn.entries[0]
            def _val(attr):
                return str(e[attr].value) if attr in e else None
            try:
                uac = int(e["userAccountControl"].value) if "userAccountControl" in e else 0
            except Exception:
                uac = 0
            enabled = not bool(uac & ADS_UF_ACCOUNTDISABLE)
            does_not_require_preauth = bool(uac & ADS_UF_DONT_REQUIRE_PREAUTH)
            use_des_key_only = bool(uac & ADS_UF_USE_DES_KEY_ONLY)
            pwd_never_expires = bool(uac & ADS_UF_DONT_EXPIRE_PASSWD)

            rows.append(dict(
                Source="OnPrem", Scope="OnPrem", RoleOrGroup=g, ObjectType="user",
                DisplayName=_val("displayName"),
                UserPrincipalName=_val("userPrincipalName"),
                AccountEnabled=enabled,  # mirror enabled for easier pivots
                ObjectId=None, InheritedFromGroup=g,
                SamAccountName=_val("sAMAccountName"),
                Enabled=enabled,
                LastLogonDate=ad_time_to_dt(e["lastLogonTimestamp"].value) if "lastLogonTimestamp" in e else None,
                PasswordLastSet=ad_time_to_dt(e["pwdLastSet"].value) if "pwdLastSet" in e else None,
                DoesNotRequirePreAuth=does_not_require_preauth,
                UseDESKeyOnly=use_des_key_only,
                PasswordNeverExpires=pwd_never_expires
            ))
    rows.sort(key=lambda r: (r["RoleOrGroup"] or "", r["SamAccountName"] or ""))
    return rows

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Export On-Prem + Azure privileged admins, app secrets, and changes.")
    ap.add_argument("--mode", choices=["OP","AZ","BOTH"], default="BOTH")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--tenant", help="Tenant ID or domain")
    ap.add_argument("--auth", choices=["devicecode","clientsecret"], help="Auth method for Graph")
    ap.add_argument("--client-id")
    ap.add_argument("--client-secret")
    # On-Prem
    ap.add_argument("--ad-server", help="e.g. ldaps://dc1.example.com:636 or dc1.example.com")
    ap.add_argument("--ad-user")
    ap.add_argument("--ad-pass")
    ap.add_argument("--ad-base-dn")
    ap.add_argument("--ad-extra-groups", help="Comma-separated CNs or DNs of extra privileged groups")
    ap.add_argument("--ad-tls-skip-verify", action="store_true", help="Allow insecure LDAPS (skip cert validation)")
    # Time filters
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", help="MMDDYYYY")
    ap.add_argument("--end", help="MMDDYYYY")
    ap.add_argument("--last", type=int, help="Last N days")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # Compute window
    win_start = None
    win_end = None
    if args.all:
        pass
    elif args.last:
        win_end = datetime.now(timezone.utc)
        win_start = win_end - timedelta(days=abs(args.last))
    elif args.start or args.end:
        win_start = parse_mmddyyyy(args.start) if args.start else None
        win_end = parse_mmddyyyy(args.end) if args.end else None
        if win_start and win_end and win_end < win_start:
            log("ERROR: --end cannot be before --start.")
            sys.exit(2)

    # Azure auth (if needed)
    token = None
    if args.mode in ("AZ","BOTH"):
        if not args.tenant or not args.auth or not args.client_id:
            log("ERROR: --tenant, --auth, and --client-id are required for Azure mode.")
            sys.exit(2)
        if args.auth == "devicecode":
            token = acquire_token_devicecode(args.tenant, args.client_id)
        else:
            if not args.client_secret:
                log("ERROR: --client-secret required for clientsecret auth.")
                sys.exit(2)
            token = acquire_token_clientsecret(args.tenant, args.client_id, args.client_secret)
        log(f"Graph token acquired. Required permissions: {NEEDED_PERMISSIONS}")

    # Collect
    all_rows: List[Dict[str, Any]] = []

    # On-Prem
    op_rows: List[Dict[str, Any]] = []
    if args.mode in ("OP","BOTH"):
        extra = [g.strip() for g in (args.ad_extra_groups or "").split(",")] if args.ad_extra_groups else None
        op_rows = collect_onprem_admins(args.ad_server, args.ad_user, args.ad_pass, args.ad_base_dn, args.ad_tls_skip_verify, extra)
        write_csv_with_header(
            os.path.join(args.outdir, "OnPrem_AD_Admins.csv"),
            op_rows,
            header=[
                "Source","Scope","RoleOrGroup","ObjectType","DisplayName","UserPrincipalName","AccountEnabled",
                "ObjectId","InheritedFromGroup","SamAccountName","Enabled","LastLogonDate","PasswordLastSet",
                "DoesNotRequirePreAuth","UseDESKeyOnly","PasswordNeverExpires",
                "DirectoryScopeId","AssignmentStart","AssignmentEnd"
            ]
        )
        all_rows.extend(op_rows)

    # Azure
    active_rows: List[Dict[str, Any]] = []
    elig_rows: List[Dict[str, Any]] = []
    if args.mode in ("AZ","BOTH"):
        active_rows, elig_rows = collect_azure_roles(token)
        write_csv_with_header(
            os.path.join(args.outdir, "AzureAD_Admins_Active.csv"),
            active_rows,
            header=[
                "Source","Scope","RoleOrGroup","ObjectType","DisplayName","UserPrincipalName","AccountEnabled",
                "ObjectId","InheritedFromGroup","SamAccountName","Enabled","LastLogonDate","PasswordLastSet",
                "DoesNotRequirePreAuth","UseDESKeyOnly","PasswordNeverExpires",
                "DirectoryScopeId","AssignmentStart","AssignmentEnd"
            ]
        )
        write_csv_with_header(
            os.path.join(args.outdir, "AzureAD_Admins_Eligible.csv"),
            elig_rows,
            header=[
                "Source","Scope","RoleOrGroup","ObjectType","DisplayName","UserPrincipalName","AccountEnabled",
                "ObjectId","InheritedFromGroup","SamAccountName","Enabled","LastLogonDate","PasswordLastSet",
                "DoesNotRequirePreAuth","UseDESKeyOnly","PasswordNeverExpires",
                "DirectoryScopeId","AssignmentStart","AssignmentEnd"
            ]
        )
        all_rows.extend(active_rows)
        all_rows.extend(elig_rows)

        # App secrets/certs
        secrets_rows = collect_app_secrets(token, active_rows, elig_rows, win_start, win_end)
        write_csv_with_header(
            os.path.join(args.outdir, "AzureAD_AppSecrets.csv"),
            secrets_rows,
            header=[
                "AppId","ApplicationDisplayName","ServicePrincipalId","ServicePrincipalDisplayName",
                "CredentialType","CredentialDisplayName","StartDateTime","EndDateTime","KeyId","Hint"
            ]
        )
        # Audit logs
        changes_rows = collect_app_secret_changes(token, win_start, win_end)
        write_csv_with_header(
            os.path.join(args.outdir, "AzureAD_AppSecretChanges.csv"),
            changes_rows,
            header=[
                "ActivityDateTime","ActivityDisplayName","InitiatedBy","Category",
                "TargetType","TargetId","AdditionalDetails"
            ]
        )

    # Merge
    all_rows_sorted = sorted(all_rows, key=lambda r: (
        r.get("Source") or "", r.get("Scope") or "", r.get("RoleOrGroup") or "", r.get("ObjectType") or "",
        r.get("UserPrincipalName") or "", r.get("DisplayName") or ""
    ))
    write_csv_with_header(
        os.path.join(args.outdir, "All_Admins.csv"),
        all_rows_sorted,
        header=[
            "Source","Scope","RoleOrGroup","ObjectType","DisplayName","UserPrincipalName","AccountEnabled",
            "ObjectId","InheritedFromGroup","SamAccountName","Enabled","LastLogonDate","PasswordLastSet",
            "DoesNotRequirePreAuth","UseDESKeyOnly","PasswordNeverExpires",
            "DirectoryScopeId","AssignmentStart","AssignmentEnd"
        ]
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
