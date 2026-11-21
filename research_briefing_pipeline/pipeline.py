import sqlite3
from datetime import datetime
from pathlib import Path
import feedparser
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "articles.db"
REPORTS_DIR = BASE_DIR / "briefings"
REPORTS_DIR.mkdir(exist_ok=True)

FEEDS = [
    "https://hnrss.org/newest?q=AI+OR+Python&points=50",
    "https://blog.langchain.com/rss/",
]

llm = ChatOllama(model="llama3.2", temperature=0.2)

class ArticleSummary(BaseModel):
    category: str = Field(description="One of: Agentic AI, Python Backend, DevOps, Industry News")
    tldr: str = Field(description="Two-sentence technical summary")
    actionable_takeaway: str = Field(description="One practical takeaway for an engineer")

# --- 1. SQLite Deduplication Layer ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_articles (
                link TEXT PRIMARY KEY,
                title TEXT,
                category TEXT,
                summary TEXT,
                created_at TEXT
            )
            """
        )

def is_seen(link: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT 1 FROM seen_articles WHERE link = ?", (link,))
        return cur.fetchone() is not None

def save_article(link: str, title: str, category: str, summary: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO seen_articles (link, title, category, summary, created_at) VALUES (?, ?, ?, ?, ?)",
            (link, title, category, summary, datetime.utcnow().isoformat()),
        )

# --- 2. Fetch, Summarize & Generate Briefing ---
def run_briefing_pipeline(max_new_articles: int = 5):
    init_db()
    structured_llm = llm.with_structured_output(ArticleSummary)
    digest_items = []

    for feed_url in FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            if len(digest_items) >= max_new_articles:
                break
            link = entry.get("link", "")
            title = entry.get("title", "Untitled")
            raw_summary = entry.get("summary", "")[:1500]

            if not link or is_seen(link):
                continue

            print(f"[PROCESSING] {title}")
            prompt = f"Analyze this engineering article:\nTitle: {title}\nContent: {raw_summary}"
            analysis: ArticleSummary = structured_llm.invoke(prompt)

            save_article(link, title, analysis.category, analysis.tldr)
            digest_items.append((title, link, analysis))

    if not digest_items:
        print("No new articles found since last run.")
        return

    today = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = REPORTS_DIR / f"briefing_{today}.md"

    lines = [f"# Automated AI & Python Engineering Briefing ({today})\n"]
    for title, link, item in digest_items:
        lines.append(f"## [{item.category}] {title}")
        lines.append(f"- **Source:** {link}")
        lines.append(f"- **Summary:** {item.tldr}")
        lines.append(f"- **Engineer Takeaway:** {item.actionable_takeaway}\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Generated briefing report at: {report_path}")

if __name__ == "__main__":
    run_briefing_pipeline()