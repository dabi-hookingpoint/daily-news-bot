import json
import operator
import os
import pathlib
from datetime import datetime
from time import mktime
from typing import Annotated, TypedDict

import feedparser
import requests
import trafilatura
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import OpenAI
from pydantic import BaseModel, Field

from charts import fetch_box_office_top10, fetch_ott_top10
from config import load_config

client = OpenAI()
UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}

BATCH, TARGET = 40, 5  # 예선 묶음 크기, 최종 발행 건수(일반 뉴스). 평론은 1건 고정.

CFG = load_config()  # 독자·기준·토픽·소스 — 분야가 바뀌면 audience.yaml만 고치면 된다
GENERAL_FEEDS = [(s.이름, s.url) for s in CFG.소스 if s.역할 == "일반"]
REVIEW_FEEDS = [(s.이름, s.url) for s in CFG.소스 if s.역할 == "평론"]


def _build_criteria(cfg) -> str:
    lines = [f"독자는 {cfg.독자.누구}입니다."]
    lines += [f"- {c}" for c in cfg.중요도_기준]
    if cfg.버릴_것:
        lines.append("버릴 것: " + ", ".join(cfg.버릴_것))
    return "\n".join(lines)


CRITERIA = _build_criteria(CFG)

REPORT_SYS = (
    f"당신은 {CFG.독자.누구}를 위한 뉴스레터 기자입니다.\n"
    "아래 기사 본문을 읽고 헤드라인·요약·왜 중요한지를 쓰세요.\n"
    "'주목된다·기대를 모은다' 같은 기자체 표현은 쓰지 마세요."
)

CHECK_SYS = (
    "아래 [원문]을 기준으로 [헤드라인]·[요약]·[왜 중요한지] 세 항목에 원문에 없는 내용이 있는지 판정하세요.\n"
    "숫자·수치·인물·기관명·인과관계를 특히 원문과 대조하세요.\n"
    "번역체 표현이나 단위 환산(예: '5억 원'과 '$500 million'은 같은 값)은 문제로 보지 마세요."
)

COLORS = {t.이름: int(t.색상.lstrip("#"), 16) for t in CFG.토픽}
DEFAULT_COLOR = 0x5F7476
REVIEW_COLOR = 0xC33C2E  # 로튼토마토 톤
CHART_COLORS = {"movie": 0x0B6E77, "ott": 0x4C7C9C}
TITLE_MAX, DESC_MAX, EMBED_MAX, TOTAL_MAX = 256, 4096, 10, 5800  # 6000에서 여유를 둔다


class Pick(BaseModel):
    index: int = Field(description="후보 목록에서의 번호")
    reason: str = Field(description="왜 골랐는지 한 문장")
    event: str = Field(description="이 기사가 다루는 사건을 짧은 라벨로. 같은 사건이면 같은 라벨")


class Shortlist(BaseModel):
    picks: list[Pick]


class Draft(BaseModel):
    headline: str = Field(description="20자 내외의 한국어 헤드라인")
    summary: str = Field(description="세 문장 요약. ~합니다체, 과장 없이 건조하게")
    why: str = Field(description="독자에게 왜 중요한지 한 문장")


class Verdict(BaseModel):
    ok: bool = Field(description="요약이 원문에 근거하면 true")
    problems: list[str] = Field(description="근거 없는 부분. 없으면 빈 목록")


class Brief(TypedDict):
    label: str          # 로그 구분용: "일반" 또는 "평론"
    feeds: list          # [(source_name, rss_url), ...] — 이 갈래가 수집할 소스
    target: int           # 최종 몇 건을 뽑을지 (일반=5, 평론=1)
    hours: int
    collected: list
    picked: list
    drafted: Annotated[list, operator.add]
    verified: list
    log: Annotated[list, operator.add]


def make_init(label: str, feeds: list, target: int) -> dict:
    return {
        "label": label,
        "feeds": feeds,
        "target": target,
        "hours": 0,
        "collected": [],
        "picked": [],
        "drafted": [],
        "verified": [],
        "log": [],
    }


def collect(s: dict) -> dict:  # ① 수집
    all_articles = []
    for name, url in s["feeds"]:
        try:
            parsed = feedparser.parse(requests.get(url, headers=UA, timeout=20).content)
        except requests.exceptions.RequestException as e:
            print(f"피드 요청 실패 {name}: {e}")
            continue
        for e in parsed.entries[:4]:
            at = None
            if getattr(e, "published_parsed", None):
                try:
                    at = datetime.fromtimestamp(mktime(e.published_parsed))
                except (TypeError, ValueError):
                    at = None
            all_articles.append(
                {
                    "source": name,
                    "title": e.title,
                    "url": getattr(e, "link", None),
                    "at": at or datetime.now(),
                }
            )
    return {
        "collected": all_articles,
        "log": [f"①[{s['label']}] 수집   {len(all_articles)}건"],
    }


def ask_picks(items, n):
    listing = "\n".join(f"{i}. [{it['source']}] {it['title']}" for i, it in enumerate(items))
    sys = (
        f"{CRITERIA}\n\n아래 목록에서 중요한 순서대로 {n}건을 고르세요.\n"
        "같은 사건을 다룬 기사에는 같은 event 라벨을 붙이세요."
    )
    out = client.chat.completions.parse(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": listing}],
        response_format=Shortlist,
    ).choices[0].message.parsed
    return [p for p in out.picks if 0 <= p.index < len(items)]


def select(s: dict) -> dict:  # ② 중요도 선별
    items = s["collected"]
    if not items:
        return {"picked": [], "log": [f"②[{s['label']}] 선별   0건 — 수집된 기사 없음"]}
    survivors = []
    for i in range(0, len(items), BATCH):
        chunk = items[i : i + BATCH]
        survivors += [chunk[p.index] for p in ask_picks(chunk, 8)]
    finals = ask_picks(survivors, s["target"]) if survivors else []
    return {
        "picked": [survivors[p.index] for p in finals],
        "log": [f"②[{s['label']}] 선별   {len(items)} → 예선 {len(survivors)} → {len(finals)}건"],
    }


def fan_report(s: dict):
    return [Send("report", {"item": it}) for it in s["picked"]]


def extract_body(url):
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded) if downloaded else None


def draft(body):
    return client.chat.completions.parse(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "system", "content": REPORT_SYS}, {"role": "user", "content": body[:6000]}],
        response_format=Draft,
    ).choices[0].message.parsed


def report(s: dict) -> dict:  # ③ 취재·요약
    it = s["item"]
    body = extract_body(it["url"])
    if not body or len(body) < 600:
        return {
            "drafted": [],
            "log": [f"   취재 제외 {it['source']} · 본문 {len(body or '')}자"],
        }
    d = draft(body)
    return {"drafted": [{**it, "body": body[:6000], **d.model_dump()}]}


def check(d):
    user = (
        f"[원문]\n{d['body'][:6000]}\n\n"
        f"[헤드라인]\n{d['headline']}\n\n"
        f"[요약]\n{d['summary']}\n\n"
        f"[왜 중요한지]\n{d['why']}"
    )
    return client.chat.completions.parse(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "system", "content": CHECK_SYS}, {"role": "user", "content": user}],
        response_format=Verdict,
    ).choices[0].message.parsed


def verify(s: dict) -> dict:  # ④ 검수
    kept, dropped = [], []
    for d in s["drafted"]:
        try:
            v = check(d)
        except Exception as e:
            dropped.append((d, [f"검수 호출 실패: {e}"]))
            continue
        if v.ok:
            kept.append(d)
        else:
            dropped.append((d, v.problems))
    detail = [f"{x['source']}({' · '.join(p) if p else '사유 없음'})" for x, p in dropped]
    return {
        "verified": kept,
        "log": [
            f"④[{s['label']}] 검수   {len(s['drafted'])} → {len(kept)}건"
            + (f" · 불합격 {detail}" if dropped else "")
        ],
    }


def build():
    g = StateGraph(Brief)
    g.add_node("collect", collect)
    g.add_node("select", select)
    g.add_node("report", report)
    g.add_node("verify", verify)
    g.add_edge(START, "collect")
    g.add_edge("collect", "select")
    g.add_conditional_edges("select", fan_report, ["report"])
    g.add_edge("report", "verify")
    g.add_edge("verify", END)
    return g


def _to_article(v: dict) -> dict:
    return {
        "headline": v["headline"],
        "summary": v["summary"],
        "why": v["why"],
        "url": v["url"],
        "source": v["source"],
        "topic": v.get("event", ""),
        "when": v["at"].strftime("%m-%d %H:%M"),
    }


def build_chart_lines(rows: list, kind: str) -> str:
    if not rows:
        return "오늘은 데이터를 가져오지 못했습니다."
    lines = []
    for r in rows[:10]:
        extra = (
            f"오늘 {r['audience_today']:,}명 · 누적 {r['audience_total']:,}명"
            if kind == "movie"
            else f"TOP10 {r['weeks_in_top10']}주째"
        )
        lines.append(f"**{r['rank']}.** [{r['title']}]({r['url']}) — {extra}")
    return "\n".join(lines)


def build_article_embed(a: dict, title_prefix: str = "", color_override: int | None = None) -> dict:
    desc = a["summary"]
    if a.get("why"):
        desc += f"\n\n💡 **{a['why']}**"
    return {
        "title": f"{title_prefix}{a['headline']}"[:TITLE_MAX],
        "description": desc[:DESC_MAX],
        "url": a["url"],
        "color": color_override if color_override is not None else COLORS.get(a.get("topic", ""), DEFAULT_COLOR),
        "footer": {"text": f"{a['source']} · {a['when']}"},
    }


def build_digest_embeds(run_id, movies, dramas, review_articles, general_articles) -> list[dict]:
    embeds = [
        {
            "title": f"🎬 {run_id} · 프로덕션 브리핑",
            "description": "박스오피스·OTT 순위, 오늘의 평론, 업계 뉴스를 모았습니다.",
            "color": DEFAULT_COLOR,
        },
        {
            "title": "🎟️ 박스오피스 TOP10",
            "description": build_chart_lines(movies, "movie"),
            "color": CHART_COLORS["movie"],
        },
        {
            "title": "📺 OTT(넷플릭스) 드라마 TOP10",
            "description": build_chart_lines(dramas, "ott"),
            "color": CHART_COLORS["ott"],
        },
    ]
    for a in review_articles:
        embeds.append(build_article_embed(a, "🍅 오늘의 평론 — ", color_override=REVIEW_COLOR))
    for a in general_articles:
        embeds.append(build_article_embed(a))

    total = lambda es: sum(
        len(e.get("title", "")) + len(e.get("description", "")) + len(e.get("footer", {}).get("text", ""))
        for e in es
    )
    while len(embeds) > EMBED_MAX or total(embeds) > TOTAL_MAX:
        embeds.pop()
    return embeds


def send(run_id, movies, dramas, review_articles, general_articles, webhook=None, dry_run=True):
    payload = {
        "username": "편집실",
        "embeds": build_digest_embeds(run_id, movies, dramas, review_articles, general_articles),
    }
    if dry_run or not webhook:
        print(
            f"[dry-run] embed {len(payload['embeds'])}개 · "
            f"{len(json.dumps(payload, ensure_ascii=False))}자 — 보내지 않음"
        )
        print("[dry-run-content] " + json.dumps(payload["embeds"], ensure_ascii=False))
        return False
    r = requests.post(webhook, json=payload, timeout=20)
    ok = r.status_code in (200, 204)
    print("발행:", "성공" if ok else f"실패 {r.status_code} {r.text[:120]}")
    return ok


def run():
    general = build().compile().invoke(make_init("일반", GENERAL_FEEDS, TARGET))
    review = build().compile().invoke(make_init("평론", REVIEW_FEEDS, 1))

    movies = fetch_box_office_top10(os.environ.get("KOBIS_API_KEY", ""))
    dramas = fetch_ott_top10()

    general_articles = [_to_article(v) for v in general["verified"]]
    review_articles = [_to_article(v) for v in review["verified"]]

    today = datetime.now().strftime("%Y-%m-%d")
    sent = send(
        today,
        movies,
        dramas,
        review_articles,
        general_articles,
        webhook=os.environ.get("DISCORD_WEBHOOK_URL"),
        dry_run=os.environ.get("DRY_RUN", "1") == "1",
    )

    log = general["log"] + review["log"] + [f"⑤ 발행   {'보냄' if sent else 'dry-run'}"]
    row = {
        "run_id": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "general_collected": len(general["collected"]),
        "general_published": len(general_articles),
        "review_collected": len(review["collected"]),
        "review_published": len(review_articles),
        "movies_charted": len(movies),
        "dramas_charted": len(dramas),
        "log": log,
    }
    path = pathlib.Path("store/metrics.jsonl")
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for line in log:
        print(line)
    return row
