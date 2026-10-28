import hmac
import json

import pandas as pd
import streamlit as st

from core import autotest, config, documents, evaluator, ingest, llm, rag, sandbox, storage
from core.i18n import t
from core.prompts import DEFAULT_PROMPTS, LANG_INFO
from views import charts


# ---------------- Tiện ích ----------------
def _flash(msg, kind="success"):
    st.session_state["_flash"] = (kind, msg)


def _show_flash():
    f = st.session_state.pop("_flash", None)
    if f:
        getattr(st, f[0])(f[1])


def _auth():
    if st.session_state.get("is_admin"):
        if st.sidebar.button(t("logout")):
            st.session_state["is_admin"] = False
            st.rerun()
        return True
    pw = st.text_input(t("password"), type="password", key="admin_pw")
    if st.button(t("login"), type="primary"):
        if hmac.compare_digest(pw.encode(), config.ADMIN_PASSWORD.encode()):
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            st.error(t("wrong_password"))
    if config.ADMIN_PASSWORD == "admin":
        st.info(t("default_pw_warn"))
    return False


def render():
    st.title( t("admin_title"))
    if not _auth():
        return
    _show_flash()
    tabs = st.tabs([t("tab_exercises"), t("tab_rag"), t("tab_prompts"),
                    t("tab_models"), t("tab_eval"), t("tab_history")])
    with tabs[0]:
        _exercises_tab()
    with tabs[1]:
        _rag_tab()
    with tabs[2]:
        _prompts_tab()
    with tabs[3]:
        _models_tab()
    with tabs[4]:
        _eval_tab()
    with tabs[5]:
        _history_tab()


# ---------------- Tab 1: Bài tập ----------------
def _parse_json_exercises(name, content):
    data = json.loads(content.decode("utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("exercises", [])
    if not isinstance(data, list):
        raise ValueError("JSON must be a list")
    items, bad = [], 0
    for raw in data:
        try:
            ex = storage.normalize_exercise(raw)
            ex["source"] = name
            items.append(ex)
        except ValueError:
            bad += 1
    return items, bad


def _llm_options(prefix):
    """Checkbox + chọn mô hình để LLM trích xuất bài từ file tự do. Trả về provider hoặc None."""
    providers = {p["name"]: p for p in storage.load_settings()["providers"] if p.get("name") and p.get("model")}
    if not providers or not st.checkbox(t("ex_llm_extract"), key=prefix + "_llm", help=t("ex_llm_extract_help")):
        return None
    return providers[st.selectbox(t("user_llm_model"), list(providers), key=prefix + "_llm_model")]


def _exercise_import_section():
    st.subheader(t("ex_import_title"))
    st.caption(t("ex_import_hint"))
    with st.expander(t("ex_doc_format")):
        st.caption(t("ex_doc_format_help"))
        st.code(documents.EXERCISE_TEMPLATE, language="text")
    files = st.file_uploader(
        t("ex_choose_file"), type=["json"] + sorted(documents.SUPPORTED_EXTENSIONS), key="ex_upload",
        accept_multiple_files=True,
    )
    if not files:
        st.session_state.pop("ex_import_state", None)
        return
    provider = _llm_options("ex")
    raw_files = [(f.name, f.getvalue()) for f in files]
    fp = ingest.fingerprint(raw_files, provider and provider["name"])
    state = st.session_state.get("ex_import_state", {})
    if state.get("fingerprint") != fp:
        items, errors, bad = [], [], 0
        with st.spinner(t("user_extracting")):
            doc_files = []
            for name, content in raw_files:
                if name.lower().endswith(".json"):
                    try:
                        found, n_bad = _parse_json_exercises(name, content)
                        items += found
                        bad += n_bad
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{name}: {e}")
                else:
                    doc_files.append((name, content))
            if doc_files:
                found, errs = ingest.load_items(doc_files, use_llm=bool(provider), provider=provider, with_solutions=True)
                items += found
                errors += errs
        state = {"fingerprint": fp, "items": items, "errors": errors, "bad": bad}
        st.session_state["ex_import_state"] = state

    for err in state["errors"]:
        st.error(t("ex_import_fail", err=err))
    items = state["items"]
    if not items:
        return
    no_solution = sum(1 for e in items if not e["solutions"])
    c1, c2, c3 = st.columns(3)
    c1.metric(t("ex_preview_found"), len(items))
    c2.metric(t("ex_with_solution"), len(items) - no_solution)
    c3.metric(t("ex_with_tests"), sum(1 for e in items if e["tests"]))
    if no_solution:
        st.warning(t("ex_preview_no_solution", n=no_solution))
    with st.expander(t("ex_preview"), expanded=True):
        st.dataframe(pd.DataFrame([{
            "id": e["id"], "title": e["title"], t("col_source"): e.get("source", ""),
            "languages": ", ".join(LANG_INFO[l]["name"] for l in e["solutions"]),
            "tests": len(e["tests"]),
        } for e in items]), hide_index=True)
        pick = st.selectbox(t("ex_select_one"), [e["id"] for e in items], key="ex_preview_pick")
        st.json(next(e for e in items if e["id"] == pick))
    if st.button(t("ex_import_btn"), type="primary", key="ex_do_import"):
        added, updated = storage.upsert_exercises([{k: v for k, v in e.items() if k != "source"} for e in items])
        rag.invalidate_kb()
        st.session_state.pop("ex_import_state", None)
        _flash(t("ex_import_ok", added=added, updated=updated, bad=state["bad"]))
        st.rerun()


def _exercises_tab():
    _exercise_import_section()

    exs = storage.load_exercises()
    n_tests = sum(1 for e in exs if e["tests"])
    c1, c2, c3 = st.columns(3)
    c1.metric(t("ex_total"), len(exs))
    c2.metric(t("ex_with_tests"), n_tests)
    c3.metric(t("ex_min_required"), f"{n_tests}/{config.MIN_EVAL_PROBLEMS}")

    if exs:
        df = pd.DataFrame([{
            "id": e["id"], "title": e["title"], "languages": ", ".join(LANG_INFO[l]["name"] for l in e["solutions"]),
            "tests": len(e["tests"]), "tags": ", ".join(e["tags"]),
        } for e in exs])
        st.dataframe(df, hide_index=True)

        with st.expander( t("ex_view")):
            pick = st.selectbox(t("ex_select_one"), [e["id"] for e in exs], key="ex_view_pick")
            st.json(next(e for e in exs if e["id"] == pick))
        with st.expander( t("ex_delete")):
            ids = st.multiselect(t("ex_select"), [e["id"] for e in exs], key="del_ids")
            if st.button(t("ex_delete_btn"), disabled=not ids):
                storage.delete_exercises(ids)
                rag.invalidate_kb()
                _flash(t("ex_deleted", n=len(ids)))
                st.rerun()
        st.download_button( t("ex_download"), json.dumps(exs, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="exercises.json", mime="application/json")

    with st.expander(t("ex_add")):
        with st.form("add_ex", clear_on_submit=True):
            title = st.text_input(t("ex_f_title"))
            prompt = st.text_area(t("ex_f_prompt"), height=100)
            tags = st.text_input(t("ex_f_tags"))
            sol_py = st.text_area(t("ex_f_py"), height=120)
            sol_cpp = st.text_area(t("ex_f_cpp"), height=120)
            sol_c = st.text_area(t("ex_f_c"), height=120)
            sol_java = st.text_area(t("ex_f_java"), height=120)
            tests_raw = st.text_area(t("ex_f_tests"), height=120,
                                     value='[{"input": "3 5\\n", "output": "8\\n"}]')
            submitted = st.form_submit_button(t("ex_add_btn"))
        if submitted:
            try:
                raw = {"title": title, "prompt": prompt, "tags": tags,
                       "solutions": {"python": sol_py, "cpp": sol_cpp, "c": sol_c, "java": sol_java},
                       "tests": json.loads(tests_raw) if tests_raw.strip() else []}
                storage.upsert_exercises([storage.normalize_exercise(raw)])
                rag.invalidate_kb()
                _flash(t("ex_added"))
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(t("ex_add_fail", err=e))


# ---------------- Tab 2: RAG / GraphRAG ----------------
def _rag_tab():
    s = storage.load_settings()
    emb, rg = s["embedding"], s["rag"]
    backends = ["tfidf", "ollama", "openai"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + t("rag_embed_cfg") + "**")
        backend = st.selectbox(t("rag_backend"), backends, index=backends.index(emb.get("backend", "tfidf")),
                               format_func=lambda b: t("backend_" + b))
        model = st.text_input(t("rag_embed_model"), emb.get("model", ""))
        base = st.text_input(t("rag_base_url"), emb.get("base_url", ""))
        key = st.text_input(t("rag_api_key"), emb.get("api_key", ""), type="password")
    with c2:
        st.markdown("**" + t("rag_params") + "**")
        top_k = st.slider(t("rag_top_k"), 1, 8, int(rg.get("top_k", 3)))
        thr = st.slider(t("rag_graph_thr"), 0.1, 0.9, float(rg.get("graph_sim_threshold", 0.35)), 0.05)
        st.caption(t("rag_graph_help"))
    if st.button(t("save")):
        s["embedding"] = {"backend": backend, "model": model, "base_url": base, "api_key": key}
        s["rag"] = {"top_k": top_k, "graph_sim_threshold": thr}
        storage.save_settings(s)
        rag.invalidate_kb()
        st.success(t("saved"))

    st.divider()
    if st.button( t("rag_build"), type="primary"):
        with st.spinner(t("rag_building")):
            try:
                st.success(t("rag_built", **rag.get_kb(force=True).stats()))
            except Exception as e:  # noqa: BLE001
                st.error(t("rag_build_fail", err=e))

    st.markdown("**" + t("rag_test") + "**")
    q = st.text_input(t("rag_test_query"), key="rag_test_q")
    mode = st.radio(t("knowledge_mode"), ["rag", "graph"], horizontal=True, format_func=lambda m: t("mode_" + m))
    if q and st.button(t("rag_test_btn")):
        try:
            hits = rag.get_kb().retrieve(q, k=top_k, mode=mode)
            if not hits:
                st.info(t("no_context"))
            for h in hits:
                ex = h["exercise"]
                with st.expander(f"{h['score']:.3f} · {ex['title']} · {t('via_' + h['via'])}"):
                    st.write(ex["prompt"])
        except Exception as e:  # noqa: BLE001
            st.error(t("rag_failed", err=e))


# ---------------- Tab 3: Prompt ----------------
def _reset_prompts():
    s = storage.load_settings()
    s["prompts"] = dict(DEFAULT_PROMPTS)
    storage.save_settings(s)
    st.session_state["prompt_user"] = DEFAULT_PROMPTS["user"]
    st.session_state["prompt_eval"] = DEFAULT_PROMPTS["eval"]


def _prompts_tab():
    s = storage.load_settings()
    st.caption(t("prompt_help"))
    st.session_state.setdefault("prompt_user", s["prompts"]["user"])
    st.session_state.setdefault("prompt_eval", s["prompts"]["eval"])
    st.text_area(t("prompt_user"), height=300, key="prompt_user")
    st.text_area(t("prompt_eval"), height=260, key="prompt_eval")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button(t("save"), key="save_prompts"):
        s["prompts"] = {"user": st.session_state["prompt_user"], "eval": st.session_state["prompt_eval"]}
        storage.save_settings(s)
        st.success(t("saved"))
    c2.button( t("reset_default"), on_click=_reset_prompts)


# ---------------- Tab 4: Mô hình ----------------
def _models_tab():
    s = storage.load_settings()
    st.caption(t("models_help"))
    df = pd.DataFrame(s["providers"], columns=["name", "type", "model", "base_url", "api_key", "rpm", "think"])
    df["think"] = df["think"].fillna("auto").replace("", "auto")
    df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce").fillna(0)
    edited = st.data_editor(
        df, num_rows="dynamic", key="prov_editor",
        column_config={
            "type": st.column_config.SelectboxColumn(options=llm.PROVIDER_TYPES, required=True),
            "api_key": st.column_config.TextColumn(help=t("models_key_help")),
            "rpm": st.column_config.NumberColumn(min_value=0, step=1, help=t("models_rpm_help")),
            "think": st.column_config.SelectboxColumn(options=["auto", "off", "on"], help=t("models_think_help")),
        },
    )
    if st.button( t("save"), key="save_models"):
        rows = edited.fillna("").to_dict("records")
        for r in rows:
            r["rpm"] = float(r.get("rpm") or 0)
            r["think"] = r.get("think") or "auto"
        rows = [r for r in rows if str(r["name"]).strip() and str(r["model"]).strip()]
        if len({r["name"] for r in rows}) != len(rows):
            st.error(t("models_dup"))
        else:
            s["providers"] = rows
            storage.save_settings(s)
            st.success(t("saved"))

    st.divider()
    names = [p["name"] for p in s["providers"]]
    if names:
        pick = st.selectbox(t("models_test_pick"), names, key="test_pick")
        if st.button(t("models_test_btn")):
            prov = next(p for p in s["providers"] if p["name"] == pick)
            with st.spinner("..."):
                text, err = llm.generate(prov, "Reply with the single word OK.", timeout=60, max_tokens=256)
            st.error(err) if err else st.success(t("models_test_ok", text=text[:100]))


# ---------------- Tab 5: Đánh giá pass@k ----------------
def _eval_document_problems(providers):
    """Đề thi lấy từ file tải lên. Trả về (danh sách bài, có_file). Bài chưa có test có thể được sinh test tự động."""
    files = st.file_uploader(
        t("eval_upload"), type=sorted(documents.SUPPORTED_EXTENSIONS), key="eval_document",
        accept_multiple_files=True, help=t("eval_upload_help"),
    )
    if not files:
        st.session_state.pop("eval_doc_state", None)
        st.session_state.pop("eval_doc_prep", None)
        return [], False
    provider = _llm_options("ev")
    raw_files = [(f.name, f.getvalue()) for f in files]
    fp = ingest.fingerprint(raw_files, provider and provider["name"])
    state = st.session_state.get("eval_doc_state", {})
    if state.get("fingerprint") != fp:
        with st.spinner(t("user_extracting")):
            items, errors = ingest.load_items(raw_files, use_llm=bool(provider), provider=provider, with_solutions=True)
        state = {"fingerprint": fp, "items": items, "errors": errors}
        st.session_state["eval_doc_state"] = state
        st.session_state.pop("eval_doc_prep", None)
    for err in state["errors"]:
        st.error(t("eval_upload_failed", err=err))
    items = state["items"]
    if not items:
        return [], True
    with_tests = [e for e in items if e["tests"]]
    st.success(t("eval_upload_loaded", n=len(items), tests=len(with_tests)))

    # Bài chưa có test: sinh tự động bằng mô hình tham chiếu
    lacking = [e for e in items if not e["tests"]]
    prep = st.session_state.get("eval_doc_prep")
    if lacking and providers:
        with st.container(border=True):
            st.markdown("**" + t("autotest_title", n=len(lacking)) + "**")
            st.caption(t("autotest_help"))
            c1, c2, c3 = st.columns([2, 1, 1])
            ref_name = c1.selectbox(t("autotest_model"), list(providers), key="ev_ref_model")
            count = c2.number_input(t("autotest_count"), 3, 20, 8, key="ev_ref_count")
            workers = c3.number_input(t("autotest_workers"), 1, 8, 2, key="ev_ref_workers")
            if st.button(t("autotest_btn"), key="ev_ref_btn"):
                bar, status = st.progress(0.0), st.empty()

                def cb(done, total, title):
                    bar.progress(done / max(total, 1))
                    status.text(f"{done}/{total} · {title}")

                ready, failed = autotest.add_tests(items, providers[ref_name], count=int(count),
                                                   workers=int(workers), progress_cb=cb)
                bar.empty()
                status.empty()
                prep = {"fingerprint": fp, "items": ready, "failed": failed, "model": ref_name}
                st.session_state["eval_doc_prep"] = prep
    if prep and prep.get("fingerprint") == fp:
        items = prep["items"]
        n_auto = sum(1 for e in items if e.get("auto_tests"))
        st.success(t("autotest_done", ok=n_auto, bad=len(prep["failed"]), model=prep["model"]))
        for pid, reason in prep["failed"].items():
            st.caption(f"⚠️ {pid}: {reason}")
    return items, True


def _eval_tab():
    s = storage.load_settings()
    providers = {p["name"]: p for p in s["providers"] if p.get("name") and p.get("model")}
    uploaded_problems, from_upload = _eval_document_problems(providers)
    source = uploaded_problems if from_upload else storage.load_exercises()
    problems_all = [e for e in source if e["tests"]]
    if from_upload and not problems_all:
        st.error(t("eval_upload_no_tests"))
    tc = sandbox.check_toolchain()

    st.caption(t("eval_help", k=config.PASS_K, target=int(config.PASS_TARGET * 100), minp=config.MIN_EVAL_PROBLEMS))
    missing = [LANG_INFO[lang]["name"] for lang in LANG_INFO if not tc.get(lang)]
    if missing:
        st.warning(t("eval_toolchain_missing", langs=", ".join(missing)))
    if len(problems_all) < config.MIN_EVAL_PROBLEMS:
        st.warning(t("eval_few_problems", n=len(problems_all), minp=config.MIN_EVAL_PROBLEMS))
    if not problems_all:
        return

    sel = st.multiselect(t("eval_models"), list(providers), default=list(providers)[:1], key="ev_models")
    c1, c2, c3 = st.columns(3)
    with c1:
        language = st.selectbox(t("prog_lang"), list(LANG_INFO), format_func=lambda l: LANG_INFO[l]["name"], key="ev_lang")
        n = st.slider(t("eval_n"), 1, 20, config.PASS_K, key="ev_n")
    with c2:
        temperature = st.slider(t("temperature"), 0.0, 1.5, 0.8, 0.1, key="ev_temp")
        modes = st.multiselect(t("eval_modes"), ["none", "rag", "graph"], default=["none"], key="ev_modes",
                               format_func=lambda m: t("mode_" + m), help=t("eval_modes_help"))
    with c3:
        uses_rag = any(m != "none" for m in modes)
        top_k = st.slider(t("rag_top_k"), 1, 6, int(s["rag"].get("top_k", 3)), key="ev_topk", disabled=not uses_rag)
        test_timeout = st.number_input(t("eval_test_timeout"), 1, 60, 5, key="ev_to")
        exclude_self = st.checkbox(t("eval_exclude_self"), True, key="ev_excl", disabled=not uses_rag,
                                   help=t("eval_exclude_help"))
    workers = st.slider(t("eval_workers"), 1, 8, 1, key="ev_workers", help=t("eval_workers_help"))
    max_p = st.number_input(t("eval_max_problems"), 0, len(problems_all), 0, key="ev_max", help=t("eval_max_help"))

    if n < config.PASS_K:
        st.warning(t("eval_n_small", k=config.PASS_K))
    combos = [(name, m) for name in sel for m in modes]
    if combos:
        st.caption(t("eval_runs_note", runs=len(combos)))

    if st.button(t("eval_run"), type="primary", disabled=not combos):
        problems = problems_all[: int(max_p)] if max_p else problems_all
        results = {}
        bar, status = st.progress(0.0), st.empty()
        for pi, (name, mode) in enumerate(combos):
            tag = f"{name} [{t('modeshort_' + mode)}]"

            def cb(i, total, msg, pi=pi, tag=tag):
                bar.progress(min(1.0, (pi + i / max(total, 1)) / len(combos)))
                status.text(f"[{tag}] {msg}")
            res = evaluator.evaluate(
                problems, providers[name], language, n=n, ks=(1, config.PASS_K), temperature=temperature,
                rag_mode=mode, top_k=top_k, template=s["prompts"]["eval"], test_timeout=int(test_timeout),
                exclude_self=exclude_self, progress_cb=cb, workers=workers)
            res["provider"] = name
            res["auto_tests"] = sum(1 for p in problems if p.get("auto_tests"))
            res["source"] = "upload" if from_upload else "bank"
            storage.append_history(res)
            results[tag if len(modes) > 1 else name] = res
        bar.progress(1.0)
        status.text(t("eval_done"))
        st.session_state["last_eval"] = results

    results = st.session_state.get("last_eval")
    if results:
        st.divider()
        _show_results(results, "last")


def _verdict(res):
    key = f"pass@{config.PASS_K}"
    val = res["summary"].get(key)
    if val is None:
        return None
    return val >= config.PASS_TARGET and res["num_problems"] >= config.MIN_EVAL_PROBLEMS


def _comparison_chart(results, ns):
    """Biểu đồ pass@k giữa các mô hình và giữa các chế độ none / RAG / GraphRAG (+ độ chênh so với none)."""
    if len(results) < 2:
        return
    ks = sorted({k for r in results.values() for k in r["ks"]})
    k = st.selectbox(t("eval_chart_k"), ks, index=len(ks) - 1, format_func=lambda x: f"pass@{x}", key=f"chart_k_{ns}")
    rows = [{"model": r.get("provider", name), "mode": r.get("rag_mode", "none"),
             "value": r["summary"].get(f"pass@{k}", 0.0) * 100} for name, r in results.items()
            if f"pass@{k}" in r["summary"]]
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = [m for m in ("none", "rag", "graph") if m in set(df["mode"])]
    chart_df = pd.DataFrame({"group": df["model"], "series": df["mode"].map(lambda m: t("modeshort_" + m)),
                             "value": df["value"]})
    charts.grouped_bar(chart_df, f"pass@{k} (%)", series_order=[t("modeshort_" + m) for m in order],
                       group_order=list(dict.fromkeys(df["model"])), y_max=100)
    if "none" in order and len(order) > 1:  # độ chênh so với LLM thuần = ảnh hưởng của RAG/GraphRAG lên pass@k
        base = df[df["mode"] == "none"].set_index("model")["value"]
        delta = df[df["mode"] != "none"].copy()
        delta["base"] = delta["model"].map(base)
        delta = delta.dropna(subset=["base"])
        if not delta.empty:
            delta["delta"] = delta["value"] - delta["base"]
            st.markdown("**" + t("eval_delta_title", k=k) + "**")
            st.dataframe(pd.DataFrame({
                t("col_model"): delta["model"], t("knowledge_mode"): delta["mode"].map(lambda m: t("modeshort_" + m)),
                "none": delta["base"].round(1), "pass@%d" % k: delta["value"].round(1),
                t("col_delta"): delta["delta"].round(1)}), hide_index=True)


def _show_results(results, ns):
    """results: {tên_model: kết quả}. Hiển thị bảng so sánh + chi tiết từng model."""
    st.subheader(t("eval_results"))
    _comparison_chart(results, ns)
    table = []
    for name, r in results.items():
        row = {t("col_model"): name, t("knowledge_mode"): t("modeshort_" + r.get("rag_mode", "none")),
               t("col_lang"): LANG_INFO[r["language"]]["name"],
               t("col_problems"): r["num_problems"], t("col_n"): r["n"]}
        for k in r["ks"]:
            row[f"pass@{k}"] = f"{r['summary'][f'pass@{k}'] * 100:.1f}%"
        v = _verdict(r)
        row[t("col_verdict")] = "—" if v is None else ("✅" if v else "❌")
        row[t("col_time")] = f"{r['seconds']}s"
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True)

    name = st.selectbox(t("eval_detail_of"), list(results), key=f"detail_{ns}")
    r = results[name]
    key = f"pass@{config.PASS_K}"
    cols = st.columns(len(r["ks"]) + 1)
    for col, k in zip(cols, r["ks"]):
        col.metric(f"pass@{k}", f"{r['summary'][f'pass@{k}'] * 100:.1f}%")
    cols[-1].metric(t("col_api_errors"), r["api_errors"])

    if r.get("auto_tests"):
        st.warning(t("eval_auto_tests_warn", n=r["auto_tests"]))
    if r.get("aborted"):
        st.error(t("eval_aborted", done=r["num_problems"], total=r.get("planned_problems", r["num_problems"]),
                   err=r["aborted"]))
    elif r["api_errors"]:
        st.warning(t("eval_api_warn", n=r["api_errors"]))

    v = _verdict(r)
    if v is True:
        st.success(t("eval_pass", k=config.PASS_K, target=int(config.PASS_TARGET * 100)))
    elif v is False:
        if r["summary"].get(key, 0) >= config.PASS_TARGET:
            st.warning(t("eval_pass_but_few", minp=config.MIN_EVAL_PROBLEMS))
        else:
            st.error(t("eval_fail", k=config.PASS_K, target=int(config.PASS_TARGET * 100)))
    else:
        st.info(t("eval_need_n", k=config.PASS_K))

    df = pd.DataFrame([{
        "id": x["id"], "title": x["title"], "c/n": f"{x['c']}/{x['n']}",
        **{f"pass@{k}": round(x[f"pass@{k}"], 3) for k in r["ks"]},
        t("col_statuses"): ", ".join(f"{st_}×{cnt}" for st_, cnt in
                                      pd.Series([s_["status"] for s_ in x["samples"]]).value_counts().items()),
    } for x in r["rows"]])
    st.dataframe(df, hide_index=True)
    st.download_button("CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"pass_at_k_{name}.csv", mime="text/csv", key=f"csv_{ns}_{name}")

    pick = st.selectbox(t("eval_inspect"), [x["id"] for x in r["rows"]], key=f"inspect_{ns}_{name}")
    row = next(x for x in r["rows"] if x["id"] == pick)
    if row.get("rag_error"):
        st.warning(row["rag_error"])
    if row.get("context_ids"):
        st.caption(t("eval_context") + ": " + ", ".join(row["context_ids"]))
    for i, smp in enumerate(row["samples"], 1):
        icon = "✅" if smp["status"] == "passed" else "❌"
        with st.expander(f"{icon} #{i} — {smp['status']}"):
            if smp["message"]:
                st.caption(smp["message"])
            st.code(smp["code"] or "(empty)", language=LANG_INFO[r["language"]]["fence"])


# ---------------- Tab 6: Lịch sử ----------------
def _history_tab():
    hist = storage.load_history()
    if not hist:
        st.info(t("hist_empty"))
        return
    labels = []
    for i, r in enumerate(hist):
        key = f"pass@{config.PASS_K}"
        v = r["summary"].get(key)
        labels.append(f"#{i + 1} · {r['timestamp']} · {r['provider']} · {LANG_INFO[r['language']]['name']} · "
                      f"{key}={'n/a' if v is None else f'{v * 100:.1f}%'} · {r['rag_mode']}")
    idx = st.selectbox(t("hist_pick"), range(len(hist)), index=len(hist) - 1, format_func=lambda i: labels[i])
    _show_results({hist[idx]["provider"]: hist[idx]}, f"hist{idx}")
    if st.button("🗑️ " + t("hist_clear")):
        storage.clear_history()
        st.rerun()
