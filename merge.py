import os
import io
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="多表合并工具", layout="wide")

RULES_FILE = "rules.json"


# ========== 工具函数 ==========
def load_rules():
    """加载规则文件并过滤掉旧格式"""
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                rules = json.load(f)
            # 兼容旧格式，确保有 weights
            for r in rules:
                if "weights" not in r:
                    r["weights"] = [1] * len(r["cols"])
            return rules
        except json.JSONDecodeError:
            return []
    return []


def save_rules(rules):
    """保存规则到文件"""
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)


def preview_file(file):
    """读取文件预览用"""
    if file.name.endswith(".csv"):
        return pd.read_csv(file, header=None)
    return pd.read_excel(file, header=None)


def read_file(file, header_row):
    """按指定表头行读取文件"""
    if file.name.endswith(".csv"):
        return pd.read_csv(file, header=header_row)
    return pd.read_excel(file, header=header_row)


def find_name_col(columns):
    """自动识别姓名列"""
    for col in columns:
        if "姓名" in str(col):
            return col
    return None


# ========== 状态初始化 ==========
if "rules" not in st.session_state:
    st.session_state.rules = load_rules()

if "small_table_info" not in st.session_state:
    st.session_state.small_table_info = []


# ========== 上传文件 ==========
st.title("📊 多个小表格合并并填充到大表格")

uploaded_files = st.file_uploader("上传小表格（多个）", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
big_table_file = st.file_uploader("上传大表格模板", type=["xlsx", "xls", "csv"])

# ========== 处理小表 ==========
if uploaded_files:
    st.subheader("📄 配置每个小表的表头行和姓名列")
    st.session_state.small_table_info = []

    for idx, file in enumerate(uploaded_files):
        with st.expander(f"📄 {file.name}", expanded=(idx == 0)):
            preview_df = preview_file(file)
            st.dataframe(preview_df.head(5))

            header_row = st.number_input(
                f"表 {idx+1} 表头行 (从0开始)", min_value=0, max_value=10, value=0, key=f"header_{idx}"
            )
            df = read_file(file, header_row)
            auto_name_col = find_name_col(df.columns)

            name_col = st.selectbox(
                f"表 {idx+1} 姓名列",
                df.columns.tolist(),
                index=df.columns.get_loc(auto_name_col) if auto_name_col in df.columns else 0,
                key=f"namecol_{idx}"
            )

            st.session_state.small_table_info.append({
                "file": file.name,
                "header_row": header_row,
                "name_col": name_col,
                "df": df
            })


# ========== 处理大表 ==========
if st.session_state.small_table_info and big_table_file:
    # 读取大表
    big_header_row = st.number_input(
        "大表表头行 (从0开始)", min_value=0, max_value=10, value=0, key="big_header"
    )
    big_df = read_file(big_table_file, big_header_row)

    auto_name_col_big = find_name_col(big_df.columns)
    big_name_col = st.selectbox(
        "大表姓名列",
        big_df.columns.tolist(),
        index=big_df.columns.get_loc(auto_name_col_big) if auto_name_col_big in big_df.columns else 0,
        key="big_namecol"
    )

    st.subheader("🛠 配置计算规则")

    table_idx = st.selectbox(
        "选择小表来源",
        options=list(range(len(st.session_state.small_table_info))),
        index=0,
        format_func=lambda i: f"表 {i+1} ({st.session_state.small_table_info[i]['file']})",
        key="table_idx"
    )

    cols = st.multiselect(
        "选择用于计算的列",
        options=[
            c for c in st.session_state.small_table_info[table_idx]["df"].columns
            if c != st.session_state.small_table_info[table_idx]["name_col"]
        ],
        key="selected_cols"
    )

    # 选择每列的运算符(+/-) 和 权重
    col_ops = []
    col_weights = []

    if cols:
        st.markdown("#### ⚖️ 设置每列的运算符和权重")
        for col in cols:
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                op = st.selectbox(f"{col} 运算符", ["+", "-"], key=f"op_{col}")
            with c2:
                w = st.number_input(f"{col} 权重", value=1.0, step=1.0, key=f"weight_{col}")
            col_ops.append(op)
            col_weights.append(w)

    target = st.selectbox("选择结果填入的大表列", big_df.columns.tolist())

    if st.button("➕ 添加规则"):
        if cols and target:
            st.session_state.rules.append({
                "table": table_idx,
                "cols": cols,
                "ops": col_ops,       # 保存每列的运算符
                "weights": col_weights,
                "target": target
            })
            save_rules(st.session_state.rules)
            st.success("✅ 规则已保存")
        else:
            st.warning("请选择列并填写目标列名")

else:
    st.info("👆 请先上传至少一个小表和一个大表，才能配置计算规则")

# ========== 显示规则 ==========
if st.session_state.rules:
    st.subheader("📋 当前规则")
    for i, r in enumerate(st.session_state.rules):
        table_name = st.session_state.small_table_info[r["table"]]["file"] \
            if r["table"] < len(st.session_state.small_table_info) else f"表 {r['table']+1}"
        cols_weights = " + ".join([f"{c}×{w}" for c, w in zip(r["cols"], r["weights"])])
        st.write(f"{i+1}. {table_name}: {cols_weights} → {r['target']}")

        if st.button(f"🗑 删除第{i+1}条规则", key=f"del_{i}"):
            st.session_state.rules.pop(i)
            save_rules(st.session_state.rules)
            st.rerun()

    if st.button("🗑️ 清空所有规则"):
        st.session_state.rules = []
        save_rules([])
        st.rerun()


# ========== 生成结果 ==========
if st.session_state.rules and st.button("🚀 生成结果"):
    # ✅ 确保 big_df 存在
    if "big_df" not in locals():
        st.error("请先上传并读取大表文件")
    else:
        result_df = big_df.copy()

        # 统一姓名列格式
        def normalize_names(series):
            return (
                series.astype(str)
                .str.strip()
                .str.replace("\u3000", " ")
                .str.replace("\xa0", " ")
            )

        result_df[big_name_col] = normalize_names(result_df[big_name_col])

        for rule in st.session_state.rules:
            try:
                info = st.session_state.small_table_info[rule["table"]]
                df = info["df"].copy()
                name_col = info["name_col"]
                df[name_col] = normalize_names(df[name_col])

                grouped = df.groupby(name_col)
                values = pd.Series(0.0, index=df[name_col].unique())

                # ✅ 按运算符和权重计算
                for col, op, w in zip(rule["cols"], rule["ops"], rule["weights"]):
                    col_sum = grouped[col].sum() * w
                    col_sum = col_sum.reindex(values.index, fill_value=0)
                    values = values + col_sum if op == "+" else values - col_sum

                values_df = values.reset_index()
                values_df.columns = [name_col, rule["target"]]

                # ✅ 调试输出一次
                st.write(f"🔍 规则 {rule['target']} 计算结果：", values_df.head())

                mapping = dict(zip(values_df[name_col], values_df[rule["target"]]))

                # ✅ 只对姓名非空的行赋值
                name_mask = big_df[big_name_col].notna() & (
                    big_df[big_name_col].astype(str).str.strip() != ""
                )
                result_df.loc[name_mask, rule["target"]] = (
                    result_df.loc[name_mask, big_name_col].map(mapping)
                )

                # ✅ 赋值完成后，强制该列为数值类型
                result_df[rule["target"]] = pd.to_numeric(result_df[rule["target"]], errors="coerce")


            except Exception as e:
                st.error(f"规则 {rule['target']} 计算失败：{e}")

        # ✅ 转换类型，避免数字列变成文本
        result_df = result_df.convert_dtypes()

        # ✅ 处理空值：数字列保持数字，字符串列替换为空白
        for col in result_df.columns:
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].astype("Float64")
            else:
                result_df[col] = result_df[col].astype(str).replace({"nan": "", "<NA>": ""})

        st.subheader("📥 下载结果")
        output = io.BytesIO()
        result_df.to_excel(output, index=False)
        st.download_button(
            "⬇️ 点击下载合并结果",
            data=output.getvalue(),
            file_name="合并结果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
