import requests
import csv
from io import StringIO

def fetch_bls_occupations():
    print("Fetching BLS occupation data...")
    
    occupations = [
        {"code": "15-1243", "title": "Database Architects"},
        {"code": "15-1242", "title": "Database Administrators"},
        {"code": "15-1252", "title": "Software Developers"},
        {"code": "15-1211", "title": "Computer Systems Analysts"},
        {"code": "15-2051", "title": "Data Scientists"},
        {"code": "15-1299", "title": "Computer Occupations (All Other)"},
        {"code": "15-1221", "title": "Computer and Information Research Scientists"},
        {"code": "15-2041", "title": "Statisticians"},
        {"code": "15-1244", "title": "Network and Computer Systems Administrators"},
        {"code": "15-1232", "title": "Computer User Support Specialists"},
    ]

    jobs = []
    for occ in occupations:
        code = occ["code"].replace("-", "")
        url = f"https://www.bls.gov/oes/current/oes{code}.htm"
        print(f"Fetching {occ['title']}...")

        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            text = r.text

            salary = "N/A"
            employment = "N/A"

            if "median annual wage" in text.lower():
                idx = text.lower().find("median annual wage")
                snippet = text[idx:idx+200]
                import re
                numbers = re.findall(r'\$[\d,]+', snippet)
                if numbers:
                    salary = numbers[0]

            jobs.append({
                "occupation_code": occ["code"],
                "title": occ["title"],
                "median_annual_salary": salary,
                "source": "BLS OEWS May 2024",
                "url": url
            })
            print(f"  {occ['title']}: {salary}")

        except Exception as e:
            print(f"  Error: {e}")

    return jobs

def fetch_bls_api():
    print("\nFetching BLS time series salary data...")
    
    series = [
        ("OEUN000000015124200008", "Database Administrators"),
        ("OEUN000000015125200008", "Software Developers"),
        ("OEUN000000015205100008", "Data Scientists"),
        ("OEUN000000015122100008", "Computer Systems Analysts"),
    ]

    jobs = []
    series_ids = [s[0] for s in series]
    
    payload = {
        "seriesid": series_ids,
        "startyear": "2022",
        "endyear": "2024"
    }
    
    r = requests.post(
        "https://api.bls.gov/publicAPI/v1/timeseries/data/",
        json=payload,
        headers={"Content-type": "application/json"}
    )
    
    data = r.json()
    
    for i, series_data in enumerate(data.get("Results", {}).get("series", [])):
        title = series[i][1] if i < len(series) else "Unknown"
        for point in series_data.get("data", [])[:3]:
            jobs.append({
                "title": title,
                "year": point.get("year"),
                "period": point.get("periodName"),
                "value": point.get("value"),
                "source": "BLS API"
            })
            print(f"  {title} ({point.get('year')}): {point.get('value')}")
    
    return jobs

def save_to_csv(jobs, filename):
    if not jobs:
        print(f"No data to save to {filename}")
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=jobs[0].keys())
        writer.writeheader()
        writer.writerows(jobs)
    print(f"Saved {len(jobs)} records to {filename}")

def main():
    occupations = fetch_bls_occupations()
    save_to_csv(occupations, "bls_occupations.csv")
    
    salary_trends = fetch_bls_api()
    save_to_csv(salary_trends, "bls_salary_trends.csv")
    
    print(f"\nDone. Files saved:")
    print("  bls_occupations.csv")
    print("  bls_salary_trends.csv")

main()