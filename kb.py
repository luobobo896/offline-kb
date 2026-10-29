#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地企业知识库 Demo（纯离线，零第三方依赖）
- 文档分块 + 中文二元组分词
- 混合检索：BM25 关键词 + TF-IDF 向量余弦（双路召回）
- 文档级权限：按 audience 过滤
- 回答：默认抽取式（带引用）；配置 LLM 后自动切换为生成式

用法：
  python3 kb.py ingest      # 建立索引
  python3 kb.py serve       # 启动服务 http://127.0.0.1:8787
  LLM 可选环境变量：KB_LLM_URL / KB_LLM_KEY / KB_LLM_MODEL
"""

import json
import math
import os
import re
import sqlite3
import sys
import urllib.request
from html import escape as html_escape
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE, "docs")
DB = os.path.join(BASE, "index.sqlite3")
PORT = int(os.environ.get("KB_PORT", "8787"))

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def tokenize(text):
    text = text.lower()
    toks = WORD_RE.findall(text)
    for run in CJK_RE.findall(text):
        if len(run) == 1:
            toks.append(run)
        else:
            for i in range(len(run) - 1):
                toks.append(run[i:i + 2])
    return toks


def parse_doc(path):
    raw = open(path, encoding="utf-8").read()
    meta = {"title": os.path.splitext(os.path.basename(path))[0], "audience": "all"}
    body = raw
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.S)
    if m:
        body = raw[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, body


def chunk_doc(meta, body, fname):
    chunks = []
    crumbs = []          # [(level, text)]
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            heading = " / ".join(t for _, t in crumbs) or meta["title"]
            chunks.append({
                "doc": fname,
                "title": meta["title"],
                "heading": heading,
                "audience": meta.get("audience", "all"),
                "text": text,
            })
        buf.clear()

    for line in body.splitlines():
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            flush()
            level = len(h.group(1))
            crumbs = [(lv, t) for lv, t in crumbs if lv < level]
            crumbs.append((level, h.group(2).strip()))
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
        if sum(len(x) for x in buf) > 420:
            flush()
    flush()
    return chunks


def cmd_ingest():
    files = sorted(f for f in os.listdir(DOCS) if f.endswith((".md", ".txt")))
    if not files:
        print("docs 目录没有文档")
        return
    conn = sqlite3.connect(DB)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""CREATE TABLE chunks(
        id INTEGER PRIMARY KEY, doc TEXT, title TEXT, heading TEXT,
        audience TEXT, text TEXT, ntok INTEGER)""")
    total = 0
    for f in files:
        meta, body = parse_doc(os.path.join(DOCS, f))
        for c in chunk_doc(meta, body, f):
            n = len(tokenize(c["text"]))
            conn.execute(
                "INSERT INTO chunks(doc,title,heading,audience,text,ntok) VALUES(?,?,?,?,?,?)",
                (c["doc"], c["title"], c["heading"], c["audience"], c["text"], n))
            total += 1
    conn.commit()
    conn.close()
    print(f"索引完成：{len(files)} 个文档，{total} 个分块 -> {DB}")


class Index:
    def __init__(self):
        conn = sqlite3.connect(DB)
        rows = conn.execute(
            "SELECT id,doc,title,heading,audience,text,ntok FROM chunks").fetchall()
        conn.close()
        self.chunks = []
        self.postings = defaultdict(lambda: defaultdict(int))  # term -> id -> tf
        self.df = defaultdict(int)
        for rid, doc, title, heading, aud, text, ntok in rows:
            i = len(self.chunks)
            self.chunks.append({
                "id": rid, "doc": doc, "title": title, "heading": heading,
                "audience": aud, "text": text, "len": ntok or 1,
            })
            tf = defaultdict(int)
            for t in tokenize(text):
                tf[t] += 1
            for t, c in tf.items():
                self.postings[t][i] = c
                self.df[t] += 1
        self.N = max(len(self.chunks), 1)
        self.avgdl = sum(c["len"] for c in self.chunks) / self.N
        self.idf = {t: math.log(1 + (self.N - d + 0.5) / (d + 0.5))
                    for t, d in self.df.items()}

    def _allowed(self, chunk, role):
        if role == "admin":
            return True
        return chunk["audience"] in ("all", role)

    def search(self, query, role="all", k=5):
        qt = tokenize(query)
        if not qt:
            return []
        k1, b = 1.5, 0.75
        bm25 = defaultdict(float)
        for t in qt:
            if t not in self.postings:
                continue
            idf = self.idf.get(t, 0.0)
            for i, tf in self.postings[t].items():
                c = self.chunks[i]
                denom = tf + k1 * (1 - b + b * c["len"] / self.avgdl)
                bm25[i] += idf * tf * (k1 + 1) / denom

        # TF-IDF 向量余弦
        qvec = defaultdict(float)
        for t in qt:
            qvec[t] += 1
        qnorm = math.sqrt(sum((w * self.idf.get(t, 1.0)) ** 2 for t, w in qvec.items())) or 1.0
        cos = {}
        cand = set()
        for t in qvec:
            cand.update(self.postings.get(t, {}).keys())
        for i in cand:
            c = self.chunks[i]
            dv = defaultdict(int)
            for t in tokenize(c["text"]):
                dv[t] += 1
            dot = sum(dv.get(t, 0) * self.idf.get(t, 1.0) * qvec[t] * self.idf.get(t, 1.0) for t in qvec)
            dnorm = math.sqrt(sum((v * self.idf.get(t, 1.0)) ** 2 for t, v in dv.items())) or 1.0
            cos[i] = dot / (qnorm * dnorm)

        mb = max(bm25.values()) if bm25 else 1.0
        mc = max(cos.values()) if cos else 1.0
        qterms = set(qt)
        scored = []
        for i in set(list(bm25.keys()) + list(cos.keys())):
            c = self.chunks[i]
            if not self._allowed(c, role):
                continue
            base = 0.6 * (bm25.get(i, 0) / mb) + 0.4 * (cos.get(i, 0) / mc)
            # 查询词覆盖率：命中的不同查询词越多，越可能是正解
            cterms = set(tokenize(c["text"]))
            coverage = len(qterms & cterms) / max(len(qterms), 1)
            s = base * (0.45 + 0.55 * coverage)
            scored.append((s, coverage, c))
        scored.sort(key=lambda x: -x[0])
        # 相关性下限：关键词碰巧重合的无关片段不返回，避免「答非所问」
        kept = []
        for s, cov, c in scored:
            if s < 0.12:
                continue
            if s < 0.30 and cov < 0.30:
                continue
            kept.append((s, c))
        return [{"score": round(s, 4), **c} for s, c in kept[:k]]


IDX = None


def get_index():
    global IDX
    if IDX is None:
        if not os.path.exists(DB):
            cmd_ingest()
        IDX = Index()
    return IDX


def llm_answer(question, hits):
    url = os.environ.get("KB_LLM_URL")
    key = os.environ.get("KB_LLM_KEY")
    model = os.environ.get("KB_LLM_MODEL", "deepseek-chat")
    if not (url and key):
        return None
    ctx = "\n\n".join(f"[{i+1}] 《{h['title']}》· {h['heading']}\n{h['text']}" for i, h in enumerate(hits[:5]))
    prompt = ("你是企业内部知识库助手。只能依据下面资料回答，资料没有就说不知道，并在句末标注引用编号。\n\n"
              f"资料：\n{ctx}\n\n问题：{question}\n回答：")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("LLM 调用失败，回退抽取式：", e, file=sys.stderr)
        return None


def extractive_answer(hits):
    if not hits:
        return ("没有找到相关制度。\n\n"
                "可能有两种情况：这类内容还没有收录，或者当前角色没有查看权限。")
    top = hits[:2]
    parts = []
    for i, h in enumerate(top):
        body = re.sub(r"\s+", " ", h["text"])[:220]
        parts.append(f"[{i+1}] 《{h['title']}》· {h['heading']}：{body}")
    return "\n\n".join(parts)


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>企业知识库 Demo</title>
<style>
:root{--bg:#0b1220;--fg:#e5e7eb;--muted:#94a3b8;--line:#1e293b;--card:#111a2b;--accent:#60a5fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:24px;margin:0 0 6px}.sub{color:var(--muted);font-size:13px;margin-bottom:24px}
.badge{display:inline-block;font-size:12px;color:var(--accent);border:1px solid var(--line);
border-radius:999px;padding:3px 10px;margin-right:8px}
.box{display:flex;gap:8px;margin-bottom:18px}
input,select{background:var(--card);border:1px solid var(--line);color:var(--fg);
border-radius:10px;padding:12px 14px;font-size:15px;outline:none}
input{flex:1}button{background:var(--accent);color:#04121f;border:0;border-radius:10px;
padding:12px 18px;font-weight:600;cursor:pointer}
.answer{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:18px;margin-bottom:20px;white-space:pre-wrap}
.hit{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:10px}
.hit .t{font-weight:600;font-size:14px}.hit .m{color:var(--muted);font-size:12px;margin:3px 0 6px}
.hit .x{color:var(--muted);font-size:13px}
.s{color:var(--accent);font-size:12px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
</style></head><body><div class="wrap">
<h1>企业知识库 Demo</h1>
<div class="sub">
<span class="badge">本地部署</span><span class="badge">数据不出内网</span><span class="badge">混合检索</span>
</div>
<div class="box">
<input id="q" placeholder="试试问：出差住宿报销上限是多少？" onkeydown="if(event.key==='Enter')go()">
<select id="role"><option value="all">普通员工</option><option value="hr">HR</option>
<option value="finance">财务</option><option value="it">IT</option><option value="admin">管理员</option></select>
<button onclick="go()">提问</button>
</div>
<div id="out"></div>

<script>
window.addEventListener('load',async()=>{
  const u=new URLSearchParams(location.search);
  const q=u.get('q');
  if(!q){document.getElementById('q').focus();return;}
  const el=document.getElementById('q');
  document.getElementById('role').value=u.get('role')||'all';
  if(u.get('typing')==='1'){
    el.value='';
    for(const ch of q){el.value+=ch;await new Promise(r=>setTimeout(r,90));}
    await new Promise(r=>setTimeout(r,250));
  }else{
    el.value=q;
  }
  go();
});
async function go(){
  const q=document.getElementById('q').value.trim();
  const role=document.getElementById('role').value;
  if(!q)return;
  const out=document.getElementById('out');
  out.innerHTML='<div class="answer">检索中…</div>';
  const r=await fetch('/api/search?q='+encodeURIComponent(q)+'&role='+role);
  const d=await r.json();
  let html='<div class="answer">'+esc(d.answer)+'</div>';
  html+='<div class="sub">召回 '+d.hits.length+' 条 · 用时 '+d.ms+'ms · '+(d.mode==='llm'?'LLM 生成':'抽取式')+'</div>';
  d.hits.forEach((h,i)=>{
    html+='<div class="hit"><div class="t">['+(i+1)+'] 《'+esc(h.title)+'》 · '+esc(h.heading)+' <span class="s">score '+h.score+'</span></div>'
      +'<div class="m">来源：<a href="/doc?name='+encodeURIComponent(h.doc)+'&role='+encodeURIComponent(role)+'">'+esc(h.doc)+'</a> · 可见范围：'+esc(h.audience)+' · <span class="s">点击核对原文</span></div>'
      +'<div class="x">'+esc(h.text.slice(0,180))+'…</div></div>';
  });
  out.innerHTML=html;
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
</script></div></body></html>"""


DOC_PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title} · 原文</title>
<style>
:root{{--bg:#0b1220;--fg:#e5e7eb;--muted:#94a3b8;--line:#1e293b;--card:#111a2b;--accent:#60a5fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:820px;margin:0 auto;padding:28px 20px 80px}}
a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
h1{{font-size:22px;margin:8px 0 4px}}.meta{{color:var(--muted);font-size:13px}}
.doc{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;
white-space:pre-wrap;font-size:14px;line-height:1.85;margin-top:18px}}
</style></head><body><div class="wrap">
<div class="meta"><a href="/">← 返回问答</a></div>
<h1>{title}</h1>
<div class="meta">来源：{doc} · 可见范围：{audience} · 原文全文（未做改写）</div>
<div class="doc">{body}</div>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/stats":
            idx = get_index()
            return self._send(200, json.dumps({
                "chunks": len(idx.chunks),
                "terms": len(idx.postings),
                "docs": len({c["doc"] for c in idx.chunks}),
            }, ensure_ascii=False))
        if u.path == "/api/search":
            qs = parse_qs(u.query)
            q = (qs.get("q") or [""])[0]
            role = (qs.get("role") or ["all"])[0]
            import time
            t0 = time.time()
            hits = get_index().search(q, role=role)
            ans = llm_answer(q, hits)
            mode = "llm"
            if not ans:
                ans = extractive_answer(hits)
                mode = "extractive"
            ms = int((time.time() - t0) * 1000)
            return self._send(200, json.dumps({
                "answer": ans, "hits": hits, "ms": ms, "mode": mode,
            }, ensure_ascii=False))
        if u.path == "/doc":
            qs = parse_qs(u.query)
            name = os.path.basename((qs.get("name") or [""])[0])
            role = (qs.get("role") or ["all"])[0]
            path = os.path.join(DOCS, name)
            if not name.endswith((".md", ".txt")) or not os.path.exists(path):
                return self._send(404, "文档不存在", "text/plain; charset=utf-8")
            meta, body = parse_doc(path)
            aud = meta.get("audience", "all")
            if role != "admin" and aud not in ("all", role):
                return self._send(403, "当前角色无权查看该文档", "text/plain; charset=utf-8")
            return self._send(200, DOC_PAGE.format(
                title=html_escape(meta["title"]), doc=html_escape(name),
                audience=html_escape(aud), body=html_escape(body)),
                "text/html; charset=utf-8")
        self._send(404, "{}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "serve":
        if not os.path.exists(DB):
            cmd_ingest()
        print(f"知识库已启动：http://127.0.0.1:{PORT}")
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    else:
        print("用法：python3 kb.py [ingest|serve]")
