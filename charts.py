import csv
import io
from datetime import datetime, timedelta
from urllib.parse import quote

import requests

UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}

KOBIS_BASE = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
NETFLIX_TOP10_URL = "https://top10.netflix.com/data/all-weeks-countries.tsv"


def fetch_box_office_top10(api_key: str, target_date: str | None = None) -> list[dict]:
    """KOBIS 일별 박스오피스 TOP10. 실패하면 빈 리스트 — 검수 대상이 아닌 실제 순위 데이터라
    파이프라인 전체를 막지 않고 그냥 그 섹션만 비워둔다."""
    if not api_key:
        print("KOBIS_API_KEY가 없어 박스오피스 TOP10을 건너뜀")
        return []
    if target_date is None:  # 당일 집계는 아직 안 끝났으니 어제 날짜를 쓴다
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    try:
        r = requests.get(
            KOBIS_BASE,
            params={"key": api_key, "targetDt": target_date},
            headers=UA,
            timeout=20,
        )
        r.raise_for_status()
        rows = r.json()["boxOfficeResult"]["dailyBoxOfficeList"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"KOBIS 박스오피스 조회 실패: {e}")
        return []
    return [
        {
            "rank": int(row["rank"]),
            "title": row["movieNm"],
            "audience_today": int(row["audiCnt"]),
            "audience_total": int(row["audiAcc"]),
            "url": f"https://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieDtl.do?movieCd={row['movieCd']}",
        }
        for row in rows
    ]


def fetch_ott_top10(country: str = "South Korea", category: str = "TV") -> list[dict]:
    """Netflix 공식 Top10 데이터셋에서 최신 주차 순위. category는 'TV'(드라마) 또는 'Films'.
    Netflix는 매주 화요일에만 갱신하므로, 매일 받아도 대부분 같은 주차가 반복된다."""
    try:
        r = requests.get(NETFLIX_TOP10_URL, headers=UA, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Netflix Top10 조회 실패: {e}")
        return []

    reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
    rows = [row for row in reader if row["country_name"] == country and row["category"] == category]
    if not rows:
        return []
    latest_week = max(row["week"] for row in rows)
    rows = [row for row in rows if row["week"] == latest_week]
    rows.sort(key=lambda row: int(row["weekly_rank"]))
    return [
        {
            "rank": int(row["weekly_rank"]),
            "title": row["show_title"],
            "season": "" if row["season_title"] == "N/A" else row["season_title"],
            "weeks_in_top10": int(row["cumulative_weeks_in_top_10"]),
            "url": f"https://www.netflix.com/search?q={quote(row['show_title'])}",
        }
        for row in rows
    ]
