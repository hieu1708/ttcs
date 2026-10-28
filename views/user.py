import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from core import documents, ingest, influence, llm, rag, sandbox, storage
from core.i18n import t
from core.prompts import LANG_INFO, build_prompt
from views import charts

MAX_QUESTIONS = 20
MODES = ["none", "rag", "graph"]


def _ask(provider, prompt, temperature):
    t0 = time.time()
    text, err = llm.generate(provider, prompt, temperature=temperature, timeout=300)
    return {"name": provider["name"], "text": text, "error": err, "seconds": round(time.time() - t0, 1)}


def _mode_label(mode):
    return t("modeshort_" + mode)


# ---------------- Nạp câu hỏi từ file ----------------
def _documents_section(providers):
    """Trả về danh sách câu hỏi được chọn từ các file đã tải lên (có thể rỗng)."""
    uploaded = st.file_uploader(
        t("user_upload"), type=sorted(documents.SUPPORTED_EXTENSIONS), key="u_document",
        accept_multiple_files=True, help=t("user_upload_help"),
    )
    if not uploaded:
        st.session_state.pop("u_doc_state", None)
        return []

    use_llm = st.checkbox(t("user_llm_extract"), key="u_llm_extract", help=t("user_llm_extract_help"))
    extractor = None
    if use_llm:
        name = st.selectbox(t("user_llm_model"), list(providers), key="u_llm_extract_model")
        extractor = providers[name]

    files = [(f.name, f.getvalue()) for f in uploaded]
    fp = ingest.fingerprint(files, use_llm, extractor and extractor["name"])
    state = st.session_state.get("u_doc_state", {})
    if state.get("fingerprint") != fp:
        with st.spinner(t("user_extracting")):
            items, errors = ingest.load_items(files, use_llm=use_llm, provider=extractor, with_solutions=False)
        state = {"fingerprint": fp, "items": items, "errors": errors}
        st.session_state["u_doc_state"] = state
        st.session_state.pop("u_pick", None)

    items, errors = state["items"], state["errors"]
    for err in errors:
        st.warning(t("user_upload_failed", err=err))
    if not items:
        return []
    st.success(t("user_doc_loaded", n=len(items), f=len(files)))

    labels = {it["id"]: f"{i}. {it['title']}" for i, it in enumerate(items, 1)}
    default = [it["id"] for it in items[:5]]
    picked = st.multiselect(t("user_pick_questions"), list(labels), default=default,
                            format_func=labels.get, key="u_pick")
    if len(picked) > MAX_QUESTIONS:
        st.warning(t("user_too_many", n=MAX_QUESTIONS))
        picked = picked[:MAX_QUESTIONS]
    chosen = [it for it in items if it["id"] in picked]
    with st.expander(t("user_view_questions")):
        for it in chosen:
            st.markdown(f"**{labels[it['id']]}** · `{it['source']}`")
            st.text(it["prompt"][:1500] + (" ..." if len(it["prompt"]) > 1500 else ""))
    return chosen


# ---------------- Sinh câu trả lời ----------------
def _generate(questions, chosen, modes, language, ans_lang, top_k, temperature, settings, providers, run_examples):
    bar, status = st.progress(0.0), st.empty()
    items = []
    for qi, q in enumerate(questions):
        status.text(t("progress_question", i=qi + 1, n=len(questions), title=q["title"]))
        hits_by_mode, prompts, notes = {}, {}, []
        for mode in modes:
            hits = []
            if mode != "none":
                try:
                    hits = rag.get_kb().retrieve(q["prompt"], k=top_k, mode=mode)
                except Exception as e:  # noqa: BLE001
                    notes.append(t("rag_failed", err=e))
            hits_by_mode[mode] = hits
            prompts[mode] = build_prompt(settings["prompts"]["user"], q["prompt"], language, ans_lang, hits)

        tasks = [(m, n) for m in modes for n in chosen]
        with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as pool:
            outs = list(pool.map(lambda mn: _ask(providers[mn[1]], prompts[mn[0]], temperature), tasks))

        runs = []
        for (mode, name), out in zip(tasks, outs):
            run = {"model": name, "mode": mode, "text": out["text"], "error": out["error"],
                   "seconds": out["seconds"], "code": ""}
            if not out["error"]:
                run["code"] = llm.extract_code(out["text"], language)
                if run_examples and q.get("tests") and run["code"]:
                    run["test"] = sandbox.run_tests(run["code"], language, q["tests"])
            runs.append(run)
        for run in runs:  # % ảnh hưởng: so với code của chính mô hình đó khi không dùng RAG
            if run["mode"] == "none" or run["error"]:
                continue
            base = next((r["code"] for r in runs if r["model"] == run["model"] and r["mode"] == "none"
                         and not r["error"]), None)
            run["influence"] = influence.analyze(run["code"], hits_by_mode[run["mode"]], language, base)
        items.append({"title": q["title"], "question": q["prompt"], "hits": hits_by_mode,
                      "runs": runs, "notes": notes})
        bar.progress((qi + 1) / len(questions))
    status.empty()
    bar.empty()
    return {"language": language, "modes": modes, "models": list(chosen), "items": items}


# ---------------- Hiển thị ----------------
def _render_influence(run):
    info = run.get("influence")
    if run["mode"] == "none" or run["error"]:
        return
    if not info:
        st.caption(t("no_context"))
        return
    value = influence.headline(info)
    label = t("influence_net" if info["net_pct"] is not None else "influence_overlap", mode=_mode_label(run["mode"]))
    st.metric(label, f"{value:.1f}%")
    st.progress(min(1.0, value / 100.0))
    detail = [t("influence_overlap_short", v=f"{info['overlap_pct']:.1f}")]
    if info["change_pct"] is not None:
        detail.append(t("influence_change", v=f"{info['change_pct']:.1f}"))
    st.caption(" · ".join(detail))
    if run["mode"] == "graph":
        st.caption(t("influence_split", vec=f"{info['vector_pct']:.1f}", graph=f"{info['graph_pct']:.1f}"))
    if info["per_ref"]:
        df = pd.DataFrame([{t("col_ref"): r["title"], t("col_via"): t("via_" + r["via"]),
                            t("col_share"): round(r["pct"], 1)} for r in info["per_ref"]])
        st.dataframe(df, hide_index=True, column_config={
            t("col_share"): st.column_config.ProgressColumn(t("col_share"), format="%.1f%%", min_value=0, max_value=100)})


def _render_run(run, res):
    with st.container(border=True):
        st.markdown(f"#### {run['model']}")
        st.caption(f"{run['seconds']}s")
        if run["error"]:
            st.error(run["error"])
            return
        st.markdown(run["text"])
        test = run.get("test")
        if test:
            if test["passed"]:
                st.success(t("test_ok", ok=test["tests_passed"], total=test["tests_total"]))
            else:
                st.error(t("test_fail", status=test["status"], ok=test["tests_passed"], total=test["tests_total"]))
        with st.expander(t("copy_code")):
            st.code(run["code"], language=LANG_INFO[res["language"]]["fence"])
        _render_influence(run)


def _render_item(item, res):
    with st.expander(t("question_label"), expanded=len(res["items"]) == 1):
        st.text(item["question"][:3000])
    for note in item["notes"]:
        st.warning(note)
    modes = res["modes"]
    boxes = st.tabs([_mode_label(m) for m in modes]) if len(modes) > 1 else [st.container()]
    for mode, box in zip(modes, boxes):
        with box:
            hits = item["hits"][mode]
            if hits:
                with st.expander(t("sources") + f" ({len(hits)})"):
                    for h in hits:
                        ex = h["exercise"]
                        st.markdown(f"**{ex['title']}** — score `{h['score']:.2f}` · {t('via_' + h['via'])}")
                        st.caption(ex["prompt"])
            elif mode != "none":
                st.info(t("no_context"))
            runs = [r for r in item["runs"] if r["mode"] == mode]
            for start in range(0, len(runs), 3):
                group = runs[start:start + 3]
                for col, run in zip(st.columns(len(group)), group):
                    with col:
                        _render_run(run, res)


def _comparison_frame(res):
    rows = []
    for item in res["items"]:
        for r in item["runs"]:
            if r["error"]:
                continue
            info = r.get("influence")
            test = r.get("test")
            rows.append({
                "question": item["title"], "model": r["model"], "mode": r["mode"],
                "influence": 0.0 if r["mode"] == "none" else (influence.headline(info) if info else None),
                "overlap": info["overlap_pct"] if info else None,
                "change": info["change_pct"] if info else None,
                "vector": info["vector_pct"] if info else None,
                "graph": info["graph_pct"] if info else None,
                "seconds": r["seconds"],
                "tokens": len(influence.tokens(r["code"])),
                "test": (100.0 if test["passed"] else 0.0) if test else None,
            })
    return pd.DataFrame(rows)


def _render_comparison(res):
    df = _comparison_frame(res)
    if df.empty:
        return
    st.divider()
    st.subheader(t("compare_title"))
    metrics = ["influence", "overlap", "change", "test", "seconds", "tokens"]
    metric = st.selectbox(t("compare_metric"), metrics, format_func=lambda m: t("metric_" + m), key="u_metric")
    agg = df.groupby(["model", "mode"], sort=False).mean(numeric_only=True).reset_index()
    plot = agg.dropna(subset=[metric])
    if plot.empty:
        st.info(t("metric_unavailable"))
    else:
        chart_df = pd.DataFrame({"group": plot["model"], "series": plot["mode"].map(_mode_label),
                                 "value": plot[metric].astype(float)})
        charts.grouped_bar(chart_df, t("metric_" + metric), series_order=[_mode_label(m) for m in MODES],
                           group_order=res["models"],
                           y_max=100 if metric in ("influence", "overlap", "change", "test") else None)
    st.caption(t("influence_note"))

    if "graph" in res["modes"]:
        g = agg[agg["mode"] == "graph"].dropna(subset=["vector", "graph"])
        if not g.empty:
            st.markdown("**" + t("split_title") + "**")
            split = g.set_index("model")[["vector", "graph"]]
            split.columns = [t("via_vector"), t("via_graph")]
            st.bar_chart(split)

    table = agg.drop(columns=["vector", "graph"], errors="ignore").copy()
    table["mode"] = table["mode"].map(_mode_label)
    renamed = {"model": t("col_model"), "mode": t("knowledge_mode")}
    renamed.update({m: t("metric_" + m) for m in metrics})
    st.dataframe(table.rename(columns=renamed).round(1), hide_index=True)
    st.download_button(t("download_detail"), df.round(2).to_csv(index=False).encode("utf-8-sig"),
                       file_name="rag_comparison.csv", mime="text/csv")


def render():
    st.title(t("user_title"))
    st.caption(t("user_subtitle"))

    s = storage.load_settings()
    providers = {p["name"]: p for p in s["providers"] if p.get("name") and p.get("model")}
    if not providers:
        st.warning(t("no_models"))
        return

    doc_questions = _documents_section(providers)
    typed = st.text_area(t("user_question_extra" if doc_questions else "user_question"), height=140,
                         placeholder=t("user_placeholder"), key="u_question")

    c1, c2, c3 = st.columns(3)
    with c1:
        language = st.selectbox(t("prog_lang"), list(LANG_INFO), format_func=lambda l: LANG_INFO[l]["name"], key="u_lang")
        ans_lang = st.selectbox(t("answer_lang"), ["vi", "en"], key="u_ans_lang",
                                index=0 if st.session_state.get("lang", "vi") == "vi" else 1,
                                format_func=lambda x: {"vi": "Tiếng Việt", "en": "English"}[x])
    with c2:
        compare_all = st.checkbox(t("compare_modes"), key="u_compare", help=t("compare_modes_help"))
        mode = st.radio(t("knowledge_mode"), MODES, key="u_mode", disabled=compare_all,
                        format_func=lambda m: t("mode_" + m))
        with_baseline = st.checkbox(t("measure_influence"), True, key="u_baseline", help=t("measure_influence_help"),
                                    disabled=compare_all or mode == "none")
        top_k = st.slider(t("rag_top_k"), 1, 6, int(s["rag"].get("top_k", 3)), key="u_topk",
                          disabled=mode == "none" and not compare_all)
    with c3:
        chosen = st.multiselect(t("pick_models"), list(providers), default=list(providers)[:1], key="u_models")
        temperature = st.slider(t("temperature"), 0.0, 1.0, 0.2, 0.1, key="u_temp")
        run_examples = st.checkbox(t("run_examples"), True, key="u_run_examples", help=t("run_examples_help"))

    if compare_all:
        modes = list(MODES)
    elif mode != "none" and with_baseline:
        modes = ["none", mode]
    else:
        modes = [mode]

    questions = list(doc_questions)
    if typed.strip():
        questions.append({"title": typed.strip().splitlines()[0][:80], "prompt": typed.strip(), "tests": []})

    if questions and chosen:
        st.caption(t("cost_note", n=len(questions) * len(chosen) * len(modes)))
    if st.button(t("ask_btn"), type="primary", disabled=not questions or not chosen):
        st.session_state["u_result"] = _generate(questions, chosen, modes, language, ans_lang, top_k, temperature,
                                                 s, providers, run_examples)

    res = st.session_state.get("u_result")
    if not res:
        return
    st.subheader(t("answers"))
    items = res["items"]
    if len(items) > 1:
        tabs = st.tabs([f"{i}. {it['title'][:22]}" for i, it in enumerate(items, 1)])
        for tab, item in zip(tabs, items):
            with tab:
                _render_item(item, res)
    else:
        _render_item(items[0], res)
    _render_comparison(res)
