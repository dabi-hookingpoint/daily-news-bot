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

from config import load_config

client = OpenAI()
UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}

FEEDS = [
    ("TechCrunch", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("AI타임스", "https://www.aitimes.com/rss/allArticle.xml"),
    ("The Verge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
]

BATCH, TARGET = 40, 5  # 예선 묶음 크기, 최종 발행 건수

CFG = load_config()  # 독자·기준·토픽 — 분야가 바뀌면 audience.yaml만 고치면 된다


def _build_criteria(cfg) -> str:
    lines = [f"독자는 {cfg.독자.누구}입니다."]
    lines += [f"- {c}" for c in cfg.중요도_기준]
    if cfg.버릴_것:
        lines.append("버릴 것: " + ", ".join(cfg.버릴_것))
    return "\n".join(lines)


CRITERIA = _build_criteria(CFG)

REPORT_SYS = (
    f"당신은 {CFG.독자.누구}를 위한 AI 뉴스레터 기자입니다.\n"
    "아래 기사 본문을 읽고 헤드라인·요약·왜 중요한지를 쓰세요.\n"
    "'주목된다·기대를 모은다' 같은 기자체 표현은 쓰지 마세요."
)

CHECK_SYS = (
    "요약이 원문에서 뒷받침되는지 판정하세요.\n"
    "헤드라인과 요약만 보고 판단하고, 번역이나 단위 환산은 문제가 아닙니다."
)

COLORS = {t.이름: int(t.색상.lstrip("#"), 16) for t in CFG.토픽}
DEFAULT_COLOR = 0x5F7476
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
    why: str = Field(description="국내 개발팀에게 왜 중요한지 한 문장")


class Verdict(BaseModel):
    ok: bool = Field(description="요약이 원문에 근거하면 true")
    problems: list[str] = Field(description="근거 없는 부분. 없으면 빈 목록")


class Brief(TypedDict):
    hours: int
    collected: list
    picked: list
    drafted: Annotated[list, operator.add]
    verified: list
    log: Annotated[list, operator.add]


INIT = {"hours": 0, "collected": [], "picked": [], "drafted": [], "verified": [], "log": []}


def collect(s: dict) -> dict:  # ① 수집
    all_articles = []
    for name, url in FEEDS:
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
        "log": [f"① 수집   {len(all_articles)}건"],
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
    survivors = []
    for i in range(0, len(items), BATCH):
        chunk = items[i : i + BATCH]
        survivors += [chunk[p.index] for p in ask_picks(chunk, 8)]
    finals = ask_picks(survivors, TARGET)
    return {
        "picked": [survivors[p.index] for p in finals],
        "log": [f"② 선별   {len(items)} → 예선 {len(survivors)} → {len(finals)}건"],
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
    user = f"[원문]\n{d['body'][:5000]}\n\n[헤드라인]\n{d['headline']}\n\n[요약]\n{d['summary']}"
    return client.chat.completions.parse(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "system", "content": CHECK_SYS}, {"role": "user", "content": user}],
        response_format=Verdict,
    ).choices[0].message.parsed


def verify(s: dict) -> dict:  # ④ 검수
    kept, dropped = [], []
    for d in s["drafted"]:
        (kept if check(d).ok else dropped).append(d)
    return {
        "verified": kept,
        "log": [
            f"④ 검수   {len(s['drafted'])} → {len(kept)}건"
            + (f" · 불합격 {[x['source'] for x in dropped]}" if dropped else "")
        ],
    }


def build_embeds(run_id, lead, articles):
    if not articles:  # 조용한 날에도 한 장은 보낸다
        return [{"title": f"🗞️ {run_id}", "color": DEFAULT_COLOR, "description": "오늘은 조용합니다."}]
    embeds = [{"title": f"🗞️ {run_id} · AI 브리핑", "description": lead, "color": DEFAULT_COLOR}]
    for i, a in enumerate(articles, 1):
        desc = a["summary"]
        if a.get("why"):
            desc += f"\n\n💡 **{a['why']}**"
        embeds.append(
            {
                "title": f"{i}. {a['headline']}"[:TITLE_MAX],
                "description": desc[:DESC_MAX],
                "url": a["url"],
                "color": COLORS.get(a.get("topic", ""), DEFAULT_COLOR),
                "footer": {"text": f"{a['source']} · {a['when']}"},
            }
        )
    total = lambda es: sum(
        len(e.get("title", "")) + len(e.get("description", "")) + len(e.get("footer", {}).get("text", ""))
        for e in es
    )
    while len(embeds) > EMBED_MAX or total(embeds) > TOTAL_MAX:
        embeds.pop()
    return embeds


def send(run_id, lead, articles, webhook=None, dry_run=True):
    payload = {"username": "편집실", "embeds": build_embeds(run_id, lead, articles)}
    if dry_run or not webhook:
        print(
            f"[dry-run] embed {len(payload['embeds'])}개 · "
            f"{len(json.dumps(payload, ensure_ascii=False))}자 — 보내지 않음"
        )
        return False
    r = requests.post(webhook, json=payload, timeout=20)
    ok = r.status_code in (200, 204)
    print("발행:", "성공" if ok else f"실패 {r.status_code} {r.text[:120]}")
    return ok


def make_lead(arts):
    if not arts:
        return ""
    srcs = ", ".join(dict.fromkeys(a["source"] for a in arts))
    return f"오늘은 {len(arts)}건을 골랐습니다. ({srcs})"


def publish(s: dict) -> dict:  # ⑤ 발행
    arts = [
        {
            "headline": a["headline"],
            "summary": a["summary"],
            "why": a["why"],
            "url": a["url"],
            "source": a["source"],
            "topic": a.get("event", ""),
            "when": a["at"].strftime("%m-%d %H:%M"),
        }
        for a in s["verified"]
    ]
    today = datetime.now().strftime("%Y-%m-%d")
    sent = send(
        today,
        make_lead(arts),
        arts,
        webhook=os.environ.get("DISCORD_WEBHOOK_URL"),
        dry_run=os.environ.get("DRY_RUN", "1") == "1",
    )
    label = f"{len(arts)}건" if arts else "조용합니다"
    return {"log": [f"⑤ 발행   {label} · {'보냄' if sent else 'dry-run'}"]}


def build():
    g = StateGraph(Brief)
    g.add_node("collect", collect)
    g.add_node("select", select)
    g.add_node("report", report)
    g.add_node("verify", verify)
    g.add_node("publish", publish)
    g.add_edge(START, "collect")
    g.add_edge("collect", "select")
    g.add_conditional_edges("select", fan_report, ["report"])
    g.add_edge("report", "verify")
    g.add_edge("verify", "publish")
    g.add_edge("publish", END)
    return g


def run():
    out = build().compile().invoke(INIT)
    row = {
        "run_id": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "collected": len(out["collected"]),
        "picked": len(out["picked"]),
        "drafted": len(out["drafted"]),
        "published": len(out["verified"]),
        "hours": out["hours"],
        "by_source": {},
        "log": out["log"],
    }
    for a in out["verified"]:
        row["by_source"][a["source"]] = row["by_source"].get(a["source"], 0) + 1
    path = pathlib.Path("store/metrics.jsonl")
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for line in out["log"]:
        print(line)
    return out
