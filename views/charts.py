"""Biểu đồ dùng chung (cột nhóm theo mô hình / chế độ)."""
import streamlit as st


def grouped_bar(df, y_title, series_order=None, group_order=None, y_max=None):
    """df gồm 3 cột: group (trục x), series (màu), value. Cột nhóm: mỗi group có một cột cho từng series."""
    try:
        import altair as alt
        scale = alt.Scale(domain=[0, y_max]) if y_max else alt.Scale(zero=True)
        base = alt.Chart(df).encode(
            x=alt.X("group:N", title=None, sort=group_order, axis=alt.Axis(labelAngle=0, labelLimit=180)),
            xOffset=alt.XOffset("series:N", sort=series_order),
            y=alt.Y("value:Q", title=y_title, scale=scale),
            color=alt.Color("series:N", title=None, sort=series_order, legend=alt.Legend(orient="bottom")),
            tooltip=["group", "series", alt.Tooltip("value:Q", format=".2f")],
        )
        labels = base.mark_text(dy=-6, fontSize=11).encode(text=alt.Text("value:Q", format=".1f"))
        st.altair_chart(base.mark_bar() + labels)
        return
    except Exception:  # noqa: BLE001  (altair cũ không hỗ trợ xOffset -> dùng biểu đồ cột của Streamlit)
        pass
    pivot = df.pivot_table(index="group", columns="series", values="value")
    pivot = pivot[[c for c in (series_order or pivot.columns) if c in pivot.columns]]
    try:
        st.bar_chart(pivot, stack=False)
    except TypeError:
        st.bar_chart(pivot)
