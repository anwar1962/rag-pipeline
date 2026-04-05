import requests
import csv
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ONET_API_KEY")
BASE_URL = "https://api-v2.onetcenter.org"
headers = {"X-API-Key": API_KEY, "Accept": "application/json"}

def search_occupations(keyword):
    print(f"Searching: {keyword}")
    url = f"{BASE_URL}/online/occupations?keyword={keyword.replace(' ', '%20')}&start=1&end=10"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"  Error: {r.status_code}")
        return []
    return r.json().get("occupation", [])

def get_skills(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary/skills", headers=headers)
    if r.status_code != 200:
        return "N/A"
    skills = r.json().get("element", [])
    return ", ".join([s["name"] for s in skills[:8]])

def get_tasks(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary/tasks", headers=headers)
    if r.status_code != 200:
        return "N/A"
    tasks = r.json().get("task", [])
    return " | ".join([t.get("statement", t.get("name", t.get("title", ""))) for t in tasks[:4]])
def get_knowledge(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary/knowledge", headers=headers)
    if r.status_code != 200:
        return "N/A"
    items = r.json().get("element", [])
    return ", ".join([k["name"] for k in items[:8]])

def get_summary(code):
    r = requests.get(f"{BASE_URL}/online/occupations/{code}/summary", headers=headers)
    if r.status_code != 200:
        return "N/A"
    data = r.json()
    return data.get("description", "N/A")

def main():
    keywords = [
        "data engineer",
        "data scientist",
        "machine learning engineer",
        "software developer",
        "database administrator",
        "business intelligence",
        "AI engineer",
        "analytics engineer"
    ]

    all_jobs = []

    for keyword in keywords:
        occupations = search_occupations(keyword)
        print(f"  Found {len(occupations)} occupations")

        for occ in occupations:
            code = occ.get("code")
            title = occ.get("title")
            print(f"  Processing: {title} ({code})")

            all_jobs.append({
                "code": code,
                "title": title,
                "description": get_summary(code),
                "skills": get_skills(code),
                "knowledge": get_knowledge(code),
                "tasks": get_tasks(code),
                "bright_outlook": occ.get("tags", {}).get("bright_outlook", False),
                "keyword": keyword
            })

    if all_jobs:
        with open("onet_occupations.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_jobs[0].keys())
            writer.writeheader()
            writer.writerows(all_jobs)
        print(f"\nSaved {len(all_jobs)} occupations to onet_occupations.csv")
    else:
        print("No data collected")

main()