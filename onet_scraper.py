import requests
import csv
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ONET_API_KEY")
BASE_URL = "https://api-v2.onetcenter.org"
headers = {"X-API-Key": API_KEY, "Accept": "application/json"}

# Direct O*NET codes for tech and data occupations
TECH_OCCUPATIONS = {
    "15-2051.00": "Data Scientists",
    "15-1243.00": "Database Architects",
    "15-1242.00": "Database Administrators",
    "15-1252.00": "Software Developers",
    "15-1211.00": "Computer Systems Analysts",
    "15-1299.08": "Computer Systems Engineers",
    "15-1221.00": "Computer and Information Research Scientists",
    "15-2041.00": "Statisticians",
    "15-1244.00": "Network and Computer Systems Administrators",
    "15-1232.00": "Computer User Support Specialists",
    "15-1231.00": "Computer Network Support Specialists",
    "15-1212.00": "Information Security Analysts",
    "15-1253.00": "Software Quality Assurance Analysts",
    "15-1299.09": "Information Technology Project Managers",
    "15-2031.00": "Operations Research Analysts",
    "15-1241.00": "Computer Network Architects",
    "11-3021.00": "Computer and Information Systems Managers",
    "15-1211.01": "Health Informatics Specialists",
    "15-2051.01": "Business Intelligence Analysts",
    "15-1299.07": "Blockchain Engineers",
}

def get_skills(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary/skills", headers=headers)
    if r.status_code != 200:
        return "N/A"
    items = r.json().get("element", [])
    return ", ".join([s["name"] for s in items[:8]])

def get_knowledge(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary/knowledge", headers=headers)
    if r.status_code != 200:
        return "N/A"
    items = r.json().get("element", [])
    return ", ".join([k["name"] for k in items[:8]])

def get_tasks(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary/tasks", headers=headers)
    if r.status_code != 200:
        return "N/A"
    tasks = r.json().get("task", [])
    return " | ".join([t.get("statement", t.get("name", "")) for t in tasks[:5]])

def get_summary(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary", headers=headers)
    if r.status_code != 200:
        return "N/A"
    return r.json().get("description", "N/A")

def get_outlook(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}", headers=headers)
    if r.status_code != 200:
        return False
    tags = r.json().get("tags", {})
    return tags.get("bright_outlook", False)

def main():
    all_jobs = []

    for code, title in TECH_OCCUPATIONS.items():
        print(f"Fetching: {title} ({code})")
        try:
            all_jobs.append({
                "code": code,
                "title": title,
                "description": get_summary(code),
                "skills": get_skills(code),
                "knowledge": get_knowledge(code),
                "tasks": get_tasks(code),
                "bright_outlook": get_outlook(code),
                "keyword": "tech"
            })
            print(f"  Done")
        except Exception as e:
            print(f"  Error: {e}")

    with open("onet_occupations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_jobs[0].keys())
        writer.writeheader()
        writer.writerows(all_jobs)

    print(f"\nSaved {len(all_jobs)} tech occupations to onet_occupations.csv")

main()