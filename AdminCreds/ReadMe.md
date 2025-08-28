# AdminCreds (Python & PowerShell)

Export privileged administrators from **On‑Prem Active Directory** and **Azure AD (Entra ID)**, flatten **PIM (Active/Eligible) role assignments** including group expansion, and capture **application/service principal credentials** plus **secret/certificate change history**.

> (c) 2025 Shane D. Shook, All Rights Reserved.

---

## What these tools produce

Both versions generate the same set of CSVs:

* **OnPrem\_AD\_Admins.csv** – Users from on‑prem privileged groups (expanded recursively). Columns include: `Source, Scope, Role/Group, ObjectType, DisplayName, UserPrincipalName, AccountEnabled/Enabled, SamAccountName, LastLogonDate, PasswordLastSet, DoesNotRequirePreAuth, UseDESKeyOnly, PasswordNeverExpires, InheritedFromGroup`.
* **AzureAD\_Admins\_Active.csv** – PIM **Active** assignments (users & service principals), with transitive group expansion; includes `DirectoryScopeId, AssignmentStart, AssignmentEnd`.
* **AzureAD\_Admins\_Eligible.csv** – PIM **Eligible** assignments, same schema.
* **AzureAD\_AppSecrets.csv** – Flattened **PasswordCredentials** (secrets) and **KeyCredentials** (certs) for service principals that appear in the privileged role inventories. Date window filters by **credential validity overlap**.
* **AzureAD\_AppSecretChanges.csv** – Audit log entries for app/SP secret/cert **adds/removes/updates** within a date window.
* **All\_Admins.csv** – Merged view across on‑prem and Azure inventories.

> **Note on date filtering**: The **inventories** (Active/Eligible) are **not** filtered by date; they reflect current/total assignments available from Graph. The date window applies to **audit logs** and **credential validity**.

---

## AdminCreds.py (Python)

### Purpose & workflow

1. **Auth to Graph** (device code or client credentials via MSAL).
2. Pull PIM **Active**/**Eligible** schedule instances; expand groups transitively into users/SPs.
3. For every **service principal** in those inventories, fetch the **Application** and flatten **secrets/certs**.
4. Query **Directory Audit** logs for app secret/cert changes (filtered by window).
5. Optionally bind to **on‑prem AD** via LDAP(S) and expand standard privileged groups; collect key user flags.
6. Write the CSVs above and a merged **All\_Admins.csv**.

### Requirements

* Python 3.9+
* pip install requests msal ldap3 python-dateutil
* Azure permissions (app or delegated):

  * `RoleManagement.Read.Directory`, `Directory.Read.All`, `Group.Read.All`, `Application.Read.All`, `AuditLog.Read.All`
* For on‑prem: LDAP(S) reachability to a domain controller; a read‑only account.

### Command‑line options (high‑level)

* `--mode OP|AZ|BOTH` (default `BOTH`)
* Azure auth: `--auth devicecode|clientsecret`, `--tenant`, `--client-id`, \[`--client-secret`]
* On‑prem: `--ad-server`, `--ad-user`, `--ad-pass`, `--ad-base-dn` (optional), `--ad-extra-groups "CN=Tier0 Admins,..."`, `--ad-tls-skip-verify`
* Time window: `--all` (default) **or** `--start MMDDYYYY` `--end MMDDYYYY` **or** `--last N`
* Output: `--outdir PATH`

> **Security note**: LDAPS is **secure by default** (certificate validation on). Use `--ad-tls-skip-verify` only for controlled triage scenarios.

### Examples

```bash
# Install deps
pip install requests msal ldap3 python-dateutil

# Both, no time filter (device code auth)
python admin_creds.py --mode BOTH --outdir C:\Exports \
  --auth devicecode --tenant <TENANT_ID> --client-id <CLIENT_ID> --all \
  --ad-server ldaps://dc1.contoso.com:636 --ad-user "CONTOSO\\svc_reader" --ad-pass "***" --ad-base-dn "DC=contoso,DC=com"

# Azure‑only, last 120 days of secret/cert changes (client credentials)
python admin_creds.py --mode AZ --outdir C:\Exports \
  --auth clientsecret --tenant <TENANT_ID> --client-id <CLIENT_ID> --client-secret <SECRET> --last 120

# Explicit range (Jul 1–Aug 31, 2025)
python admin_creds.py --mode BOTH --outdir C:\Exports \
  --auth devicecode --tenant <TENANT_ID> --client-id <CLIENT_ID> \
  --start 07012025 --end 08312025 \
  --ad-server ldaps://dc1.contoso.com:636 --ad-user "CONTOSO\\svc_reader" --ad-pass "***"
```

---

## AdminCreds.ps1 (PowerShell)

### Purpose & workflow

1. Ensure modules (`ActiveDirectory`, `Microsoft.Graph`) are available and import them.
2. `Connect-MgGraph` with delegated scopes; select `v1.0` profile.
3. Pull PIM **Active/Eligible** schedule instances; expand groups transitively; fetch user/SP details.
4. For SPs in privileged roles, resolve the **Application** and flatten **secrets/certs**.
5. Pull **DirectoryAudit** events for app/SP secret/cert changes within the time window.
6. Expand on‑prem privileged groups with RSAT and collect user flags.
7. Emit all CSVs and the merged **All\_Admins.csv**.

### Requirements

* Windows PowerShell 5.1 or PowerShell 7+
* **On‑Prem**: RSAT AD module

  * `Add-WindowsCapability -Online -Name "Rsat.ActiveDirectory.DS-LDS.Tools~~~~0.0.1.0"`
* **Azure**: Microsoft Graph module

  * `Install-Module Microsoft.Graph -Scope CurrentUser`
* Delegated scopes on login: `RoleManagement.Read.Directory`, `Directory.Read.All`, `Group.Read.All`, `Application.Read.All`, `AuditLog.Read.All`

### Parameters (high‑level)

* `-Mode OP|AZ|BOTH` (default `BOTH`), `-OutDir PATH`, `-TenantId <id|domain>`
* Time window: `-All` **or** `-Start MMDDYYYY -End MMDDYYYY` **or** `-Last N`

### Examples

```powershell
# One‑time setup (elevated for RSAT)
Add-WindowsCapability -Online -Name "Rsat.ActiveDirectory.DS-LDS.Tools~~~~0.0.1.0"
Install-Module Microsoft.Graph -Scope CurrentUser

# Everything, no time filter
.\u200bAdminCreds.ps1 -Mode BOTH -OutDir "C:\Exports" -All

# Azure‑only, last 120 days of app secret changes
.​AdminCreds.ps1 -Mode AZ -OutDir "C:\Exports" -Last 120

# Both, explicit date range
.​AdminCreds.ps1 -Mode BOTH -OutDir "C:\Exports" -Start 07012025 -End 08312025
```

---

## Notes, limits, and tips

* **PIM inventories** are not time‑filtered; use the `AssignmentStart/End` fields for downstream analysis if you want to post‑filter historically.
* **Applications vs Service Principals**: Secrets/certs come from the **Application** object for the SP’s `appId`. First‑party/gallery apps may not expose an application object; those are skipped.
* **Large tenants**: Both versions reuse HTTP sessions and request only needed fields to minimize latency; Graph throttling is handled with backoff.
* **On‑prem performance**: Default recursion is safe; if your forest is very large, consider running from a DC and limiting extra groups.
* **CSV encoding**: UTF‑8; modern Excel opens these natively. If your fleet needs BOM or older ANSI, ask and we’ll add a toggle.

---

## Troubleshooting

* **Module not found**: Install RSAT/Graph modules per the commands above.
* **Graph 403 / Forbidden**: Ensure the account/app has the listed scopes/permissions and admin consent has been granted.
* **LDAPS connection issues (Python)**:

  * Verify `ldaps://host:636`, server cert trust, and firewall rules.
  * You can temporarily add `--ad-tls-skip-verify` for triage (not recommended for production).
* **Empty inventories**: Confirm the tenant actually uses PIM, and your identity can read role schedules.
* **No secret changes in window**: Broaden the window with `--last 120` (Py) or `-Last 120` (PS) and re‑run.

---

## Versioning

* Treat **AdminCreds.py** and **AdminCreds.ps1** as peers; functionality and output schemas are aligned.



