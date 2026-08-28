"""
Real Asset Scanner - AI-Powered Vulnerability Management System
-------------------------------------------------------------------
Scans software installed on THIS Windows machine, matches each program
against real CVEs from NVD (keyword search on product name), engineers
the same features your model was trained on, and saves the result as
real_assets_scored.csv - ready to be loaded by streamlit_app.py.

Run with: python scan_real_assets.py

Install dependencies:
    pip install requests pandas --break-system-packages
"""

import re
import time
import winreg
import requests
import pandas as pd
from datetime import datetime

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = None  # Paste your NVD API key here if you have one
REQUEST_DELAY = 0.6 if NVD_API_KEY else 1.5

# Rough heuristics for exposure/sensitivity - since this can't be detected
# automatically, we classify by common software categories. Feel free to
# edit these lists to better match your own setup.
INTERNET_FACING_KEYWORDS = [
    "chrome", "firefox", "edge", "opera", "brave", "zoom", "teams",
    "skype", "discord", "outlook", "thunderbird", "vpn", "steam",
]
SENSITIVE_DATA_KEYWORDS = [
    "outlook", "thunderbird", "bank", "wallet", "password", "vault",
    "onedrive", "dropbox", "drive", "office", "excel", "word",
]


# ---------------------------------------------------------------------------
# STEP 1: Read installed software from the Windows Registry
# ---------------------------------------------------------------------------

def get_installed_software():
    """
    Reads installed programs and their versions from the Windows Registry.
    Checks both 64-bit and 32-bit registry locations, since programs can
    be listed in either depending on how they were built.
    """
    registry_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]

    software_list = []
    seen_names = set()

    for hive, path in registry_paths:
        try:
            key = winreg.OpenKey(hive, path)
        except FileNotFoundError:
            continue

        for i in range(winreg.QueryInfoKey(key)[0]):
            try:
                subkey_name = winreg.EnumKey(key, i)
                subkey = winreg.OpenKey(key, subkey_name)
                name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                try:
                    version = winreg.QueryValueEx(subkey, "DisplayVersion")[0]
                except FileNotFoundError:
                    version = "unknown"

                if name and name not in seen_names:
                    seen_names.add(name)
                    software_list.append({"name": name, "version": version})
            except (FileNotFoundError, OSError):
                continue

    return software_list


def clean_product_name(name):
    """
    Strips version numbers, edition labels, and common noise from a
    product name so it works better as an NVD search keyword.
    E.g. "7-Zip 23.01" -> "7-Zip", "Microsoft Edge WebView2 Runtime" -> kept as-is.
    """
    cleaned = re.sub(r"\s+\d+(\.\d+)*\s*$", "", name)  # trailing version numbers
    cleaned = re.sub(r"\s*\(x64\)|\s*\(x86\)", "", cleaned)  # architecture tags
    return cleaned.strip()


# ---------------------------------------------------------------------------
# STEP 2: Match installed software against real CVEs via keyword search
# ---------------------------------------------------------------------------

def request_with_retry(url, params=None, max_retries=4, base_delay=2):
    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 200:
                return response
            if response.status_code in (403, 429) or 500 <= response.status_code < 600:
                wait_time = base_delay * (2 ** (attempt - 1))
                print(f"  Rate limited/error ({response.status_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            return None  # not worth retrying (e.g. 400/404)
        except requests.exceptions.RequestException:
            time.sleep(base_delay * (2 ** (attempt - 1)))
    return None


def parse_cve_record(cve):
    cve_id = cve.get("id")
    description = ""
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            description = desc.get("value", "")
            break

    metrics = cve.get("metrics", {})
    cvss_data = {}
    if "cvssMetricV31" in metrics:
        cvss_data = metrics["cvssMetricV31"][0]["cvssData"]
    elif "cvssMetricV30" in metrics:
        cvss_data = metrics["cvssMetricV30"][0]["cvssData"]

    return {
        "cve_id": cve_id,
        "description": description,
        "base_score": cvss_data.get("baseScore", None),
        "attack_vector": cvss_data.get("attackVector", "UNKNOWN"),
        "privileges_required": cvss_data.get("privilegesRequired", "UNKNOWN"),
        "user_interaction": cvss_data.get("userInteraction", "UNKNOWN"),
        "published_date": cve.get("published", None),
    }


def search_cves_for_product(product_name, max_results=5):
    """Searches NVD for CVEs whose description mentions this product name."""
    params = {"keywordSearch": product_name, "resultsPerPage": max_results}
    response = request_with_retry(NVD_API_URL, params=params)
    if response is None:
        return []

    data = response.json()
    records = []
    for item in data.get("vulnerabilities", []):
        records.append(parse_cve_record(item["cve"]))
    return records


def classify_exposure(product_name):
    name_lower = product_name.lower()
    internet_facing = any(k in name_lower for k in INTERNET_FACING_KEYWORDS)
    sensitive_data = any(k in name_lower for k in SENSITIVE_DATA_KEYWORDS)
    return internet_facing, sensitive_data


# ---------------------------------------------------------------------------
# STEP 3: Feature engineering (matches vuln_management_prototype.py)
# ---------------------------------------------------------------------------

def engineer_features(df):
    df = df.copy()
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    df["age_days"] = (datetime.now() - df["published_date"]).dt.days
    df["age_days"] = df["age_days"].fillna(0)

    vendor_counts = df["asset_name"].value_counts()
    df["vendor_frequency"] = df["asset_name"].map(vendor_counts)

    df["no_privileges_required"] = (df["privileges_required"] == "NONE").astype(int)
    df["no_user_interaction"] = (df["user_interaction"] == "NONE").astype(int)
    df["network_attack_vector"] = (df["attack_vector"] == "NETWORK").astype(int)

    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce")
    df["base_score"] = df["base_score"].fillna(df["base_score"].median())

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Step 1: Reading installed software from the registry...")
    software_list = get_installed_software()
    print(f"Found {len(software_list)} installed programs.")

    # Limit how many programs we search to keep runtime reasonable -
    # NVD rate limits make searching hundreds of programs very slow.
    # Prioritize well-known, commonly-vulnerable software first.
    priority_keywords = [
        "chrome", "firefox", "edge", "java", "python", "node", "office",
        "adobe", "vlc", "zoom", "7-zip", "winrar", "putty", "git",
        "docker", "anaconda", "visual studio", "notepad++",
    ]
    software_list.sort(
        key=lambda s: any(k in s["name"].lower() for k in priority_keywords),
        reverse=True,
    )
    software_to_scan = software_list[:25]  # adjust this number as needed

    print(f"Scanning top {len(software_to_scan)} programs against NVD...\n")

    all_matches = []
    for i, software in enumerate(software_to_scan):
        clean_name = clean_product_name(software["name"])
        print(f"[{i+1}/{len(software_to_scan)}] Searching CVEs for: {clean_name}")

        cves = search_cves_for_product(clean_name)
        internet_facing, sensitive_data = classify_exposure(clean_name)

        for cve in cves:
            all_matches.append({
                "asset_name": clean_name,
                "installed_version": software["version"],
                "internet_facing": internet_facing,
                "sensitive_data": sensitive_data,
                **cve,
            })

        time.sleep(REQUEST_DELAY)

    if not all_matches:
        print("\nNo CVE matches found. Try increasing software_to_scan or "
              "check your internet connection.")
    else:
        result_df = pd.DataFrame(all_matches)
        result_df = engineer_features(result_df)
        result_df.to_csv("real_assets_scored.csv", index=False)
        print(f"\nDone. Found {len(result_df)} CVE matches across "
              f"{result_df['asset_name'].nunique()} programs.")
        print("Saved to real_assets_scored.csv")