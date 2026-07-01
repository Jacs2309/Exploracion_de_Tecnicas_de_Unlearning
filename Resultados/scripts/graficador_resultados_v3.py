import argparse
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN VISUAL (PALETA PASTEL FRÍA)
# ──────────────────────────────────────────────────────────────
COLORES_OPONENTE = {
    "legal":  "#81D4FA",  # Celeste frío
    "random": "#A8D8B9",  # Menta frío
    "greedy": "#9FA8DA",  # Índigo pastel
    "mcs":    "#CE93D8",  # Púrpura pastel
}

COLORES_METRICA = {
    "Win":     "#A8D8B9",  
    "Draw":    "#CFD8DC",  
    "Loss":    "#9FA8DA",  
    "Forfeit": "#CE93D8",  
}

FAMILIA_COLOR = {
    "suicide":        "#F4F6F9",
    "notconnectfour": "#F2F7F4",
    "tictactoe":      "#F4F6F9",
    "connectfour":    "#F2F7F4",
}

ORDEN_OPONENTES = ["random", "legal", "greedy", "mcs"]
ORDEN_METRICAS  = ["Win", "Draw", "Loss", "Forfeit"]
METRICA_COL = {
    "Win":     "Win_Rate",
    "Draw":    "Draw_Rate",
    "Loss":    "Loss_Rate",
    "Forfeit": "Forfeit_Rate",
}

def configurar_estilo():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times New Roman", "Palatino", "serif"],
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
        "grid.color": "#e6e9ec", "grid.linewidth": 0.5, "grid.alpha": 0.8,
    })

def limpiar_nombre(nombre):
    if not isinstance(nombre, str): return str(nombre)
    return nombre.split("/")[-1].replace("|", " \n➔ ").replace("_", "-")

def nombre_archivo(nombre):
    return nombre.replace("/", "_").replace("\\", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

def fondo_juego(ax, juego):
    ax.set_facecolor(FAMILIA_COLOR.get(juego.lower(), "#Fcfcfc"))

def juego_label(juego):
    return juego.replace("notconnectfour","Not Connect 4").replace("suicide","Suicide TTT").replace("tictactoe","Tic-Tac-Toe").replace("connectfour","Connect 4").upper()

def tag_familia(ax, juego):
    familia = "Misère" if juego.lower() in ("suicide","notconnectfour") else "Classic"
    ax.text(1.01, 0.5, f"Game family:\n{familia}", transform=ax.transAxes,
            fontsize=9, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=FAMILIA_COLOR.get(juego.lower(),"#Fcfcfc"),
                      edgecolor="#cccccc", alpha=0.9))

# ──────────────────────────────────────────────────────────────
# GRÁFICO 1: BARRAS APILADAS (RENDIMIENTO)
# ──────────────────────────────────────────────────────────────
def graficar_rendimiento_apilado(df, output_dir):
    for juego in df["Game"].unique():
        for modelo in df["Model"].unique():
            df_sub = df[(df["Game"]==juego) & (df["Model"]==modelo)].copy()
            if df_sub.empty: continue

            df_sub["Opponent"] = pd.Categorical(df_sub["Opponent"], categories=ORDEN_OPONENTES, ordered=True)
            df_sub = df_sub.sort_values("Opponent").dropna(subset=["Opponent"])
            oponentes = df_sub["Opponent"].astype(str).tolist()
            if not oponentes: continue

            if "Loss_Rate" in df_sub.columns and "Forfeit_Rate" in df_sub.columns:
                df_sub["Loss_Rate"] = (df_sub["Loss_Rate"] - df_sub["Forfeit_Rate"]).clip(lower=0)

            fig, ax = plt.subplots(figsize=(6, 5.5))
            fondo_juego(ax, juego)
            
            x = np.arange(len(oponentes))
            width = 0.55
            bottom = np.zeros(len(oponentes))

            for metrica in ORDEN_METRICAS:
                col = METRICA_COL[metrica]
                vals = df_sub[col].fillna(0).values 
                
                bars = ax.bar(x, vals, width, label=metrica, 
                              color=COLORES_METRICA[metrica],
                              edgecolor="#555555", linewidth=0.7, bottom=bottom)
                
                for bar in bars:
                    h = bar.get_height()
                    if h > 0.04:
                        txt_color = "#222222" 
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + h/2,
                                f"{h:.2f}", ha="center", va="center", 
                                fontsize=9, color=txt_color, fontweight="bold")
                
                bottom += vals

            ax.set_xticks(x)
            ax.set_xticklabels([op.capitalize() for op in oponentes])
            ax.set_ylabel("Proporción de Resultados (0.0 a 1.0)", labelpad=8)
            ax.set_xlabel("Oponente GGP", labelpad=8)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", which="major", linestyle="--", alpha=0.6)
            
            ax.set_title(f"{limpiar_nombre(modelo)}\nDistribución vs Oponentes GGP — {juego_label(juego)}",
                         pad=14, fontweight="bold", fontsize=12)

            ax.legend(title="Outcome", loc="upper center",
                      bbox_to_anchor=(0.5,-0.15), ncol=4,
                      frameon=True, facecolor="white", edgecolor="#cccccc")
            tag_familia(ax, juego)

            fname = f"rendimiento_apilado_{juego}_{nombre_archivo(modelo)}.pdf"
            plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            plt.close()
            print(f"   📊 {fname}")

# ──────────────────────────────────────────────────────────────
# GRÁFICO 2: TASAS DE ERROR CON AISLAMIENTO
# ──────────────────────────────────────────────────────────────
def generar_grafico_errores(df_subset, juego, output_dir, suffix=""):
    if df_subset.empty: return
    
    METRICAS_ERR = {
        "Jugadas Ilegales": ("Illegal_Rate",      "#9FA8DA"), 
        "Error de Formato": ("Format_Error_Rate", "#81D4FA"), 
    }
    
    modelos = df_subset["Model"].unique()
    n_mod = len(modelos)
    if n_mod == 0: return
    
    n_met = len(METRICAS_ERR)
    width = 0.3
    x     = np.arange(n_mod)
    offsets = np.linspace(-(n_met-1)/2*width, (n_met-1)/2*width, n_met)

    fig, ax = plt.subplots(figsize=(max(7, n_mod*2.5), 5.5))
    fondo_juego(ax, juego)

    max_h = 0
    for i, (label, (col, color)) in enumerate(METRICAS_ERR.items()):
        vals = [df_subset[df_subset["Model"]==m][col].mean() if col in df_subset.columns else 0 for m in modelos]
        if vals: max_h = max(max_h, max(vals))
        
        bars = ax.bar(x+offsets[i], vals, width, label=label,
                      color=color, edgecolor="#555555", linewidth=0.7, alpha=0.95)
        
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="semibold", color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([limpiar_nombre(m) for m in modelos], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Tasa Promedio (0.0 a 1.0)", labelpad=8)
    ax.set_xlabel("Configuración del Modelo", labelpad=8)
    ax.set_ylim(0, max(1.1, max_h * 1.15))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.6)
    ax.set_title(f"Análisis de Robustez Sintáctica — {juego_label(juego)}",
                 pad=14, fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.25),
              ncol=2, frameon=True, facecolor="white", edgecolor="#cccccc")
    tag_familia(ax, juego)

    fname = f"errores_{juego}{suffix}.pdf"
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"   🛠️  {fname}")

def graficar_tasas_error(df, output_dir):
    juegos = df["Game"].unique()
    for juego in juegos:
        df_j = df[df["Game"]==juego].copy()
        if df_j.empty: continue
        
        if juego.lower() in ["notconnectfour", "not connect 4"]:
            modelos = df_j["Model"].unique()
            modelos_aislados = [m for m in modelos if "alpha-1" in m.lower() or "checkpoint" in m.lower()]
            modelos_base = [m for m in modelos if m not in modelos_aislados]
            
            if modelos_base:
                generar_grafico_errores(df_j[df_j["Model"].isin(modelos_base)], juego, output_dir, "_comparativo_base")
                
            for m in modelos_aislados:
                safe_suffix = "_" + nombre_archivo(m.split("➔")[-1].strip().lower())
                generar_grafico_errores(df_j[df_j["Model"] == m], juego, output_dir, safe_suffix)
        else:
            generar_grafico_errores(df_j, juego, output_dir)


# ──────────────────────────────────────────────────────────────
# GRÁFICO 3: BOXPLOTS (MEJORADOS CON ESCALA LOGARÍTMICA)
# ──────────────────────────────────────────────────────────────
def recolectar_datos_raw(raw_dir):
    time_data = []
    token_data = []
    baselines = ["random", "legal", "greedy", "mcs"]
    
    exp_files = glob.glob(os.path.join(raw_dir, "experiment_*.json"))
    experimentos_validos = {} 
    
    for ef in exp_files:
        try:
            with open(ef, 'r', encoding='utf-8') as f:
                data = json.load(f)
                exp = data.get("experiment", {})
                game = exp.get("game", "unknown").lower()
                p1 = exp.get("p1", "")
                p2 = exp.get("p2", "")
                
                results = data.get("results", [])
                if not results: continue
                
                experimentos_validos[(game, p1, p2)] = len(results)
                
                for r in results:
                    role_llm, llm_name = None, None
                    if game in ["connectfour", "notconnectfour"]:
                        if r.get("red_player") not in baselines:
                            role_llm, llm_name = "red", r.get("red_player")
                        elif r.get("black_player") not in baselines:
                            role_llm, llm_name = "black", r.get("black_player")
                    else:
                        if r.get("x_player") not in baselines:
                            role_llm, llm_name = "x", r.get("x_player")
                        elif r.get("o_player") not in baselines:
                            role_llm, llm_name = "o", r.get("o_player")
                            
                    if role_llm and llm_name:
                        tokens = r.get("tokens", {}).get(role_llm, {})
                        token_data.append({
                            "Game": game,
                            "Model": llm_name,
                            "Prompt_Tokens": tokens.get("prompt", 0),
                            "Completion_Tokens": tokens.get("completion", 0)
                        })
        except: pass

    match_files = glob.glob(os.path.join(raw_dir, "matches_*.jsonl"))
    partidas_candidatas = defaultdict(list)
    
    for mf in match_files:
        try:
            with open(mf, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    match = json.loads(line)
                    p1 = match.get("player1", "")
                    p2 = match.get("player2", "")
                    game = match.get("game", "unknown").lower()
                    key = (game, p1, p2)
                    partidas_candidatas[key].append({
                        "id_match": match.get("id_match"),
                        "timestamp": match.get("timestamp", "")
                    })
        except: pass

    valid_match_ids = set()
    for key, matches_list in partidas_candidatas.items():
        game, p1, p2 = key
        cantidad_oficial = experimentos_validos.get((game, p1, p2), 0)
        if cantidad_oficial == 0:
            cantidad_oficial = experimentos_validos.get((game, p2, p1), 0)
            
        if cantidad_oficial > 0:
            matches_list.sort(key=lambda x: x["timestamp"], reverse=True) 
            partidas_validas = matches_list[:cantidad_oficial]
            for item in partidas_validas:
                valid_match_ids.add(item["id_match"])

    move_files = glob.glob(os.path.join(raw_dir, "moves_*.jsonl"))
    for mf in move_files:
        try:
            with open(mf, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    
                    if d.get("id_match") not in valid_match_ids:
                        continue
                        
                    model = d.get("model", "unknown")
                    if model not in baselines:
                        time_data.append({
                            "Game": d.get("game", "unknown"),
                            "Model": model,
                            "Execution_Time": d.get("execution_time", 0.0)
                        })
        except: pass

    return pd.DataFrame(time_data), pd.DataFrame(token_data)

def generar_boxplot_tiempo(df_subset, juego, output_dir, suffix=""):
    if df_subset.empty: return
    # Aumentamos el tamaño de la figura (9x6 en lugar de 7x5.5) para que respire
    fig, ax = plt.subplots(figsize=(9, 6)) 
    fondo_juego(ax, juego)
    
    # width=0.6 hace las cajas un poco más gordas
    sns.boxplot(data=df_subset, x="Model_Visual", y="Execution_Time", 
                ax=ax, palette=["#A8D8B9", "#9FA8DA", "#81D4FA", "#CE93D8"], 
                linewidth=1.2, fliersize=4, width=0.6)
    
    # MAGIA: Convertir el Eje Y a escala Logarítmica
    ax.set_yscale("log")
    # Formatear el eje logarítmico para que muestre "10, 100" en lugar de "10^1, 10^2"
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    ax.set_title(f"Distribución del Tiempo de Inferencia por Turno\n{juego_label(juego)}", pad=15, fontweight="bold")
    ax.set_ylabel("Tiempo de Ejecución (segundos, Escala Log)", labelpad=10)
    ax.set_xlabel("Configuración del Modelo", labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=10)
    
    fname = f"boxplot_tiempo_{juego}{suffix}.pdf"
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"   ⏱️  {fname}")

def generar_boxplot_tokens(df_subset, juego, output_dir, suffix=""):
    if df_subset.empty: return
    # Aumentamos el tamaño
    fig, ax = plt.subplots(figsize=(10, 6)) 
    fondo_juego(ax, juego)
    
    df_melt = pd.melt(df_subset, id_vars=["Model_Visual"], 
                      value_vars=["Prompt_Tokens", "Completion_Tokens"],
                      var_name="Tipo_Token", value_name="Cantidad")
                      
    df_melt["Tipo_Token"] = df_melt["Tipo_Token"].map({
        "Prompt_Tokens": "Tokens de Contexto (Prompt)",
        "Completion_Tokens": "Tokens Generados (Completion)"
    })
    
    sns.boxplot(data=df_melt, x="Model_Visual", y="Cantidad", hue="Tipo_Token",
                ax=ax, palette=["#9FA8DA", "#81D4FA"], linewidth=1.2, fliersize=4, width=0.7)
    
    # MAGIA: Convertir el Eje Y a escala Logarítmica
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    ax.set_title(f"Costo Computacional por Partida (Tokens)\n{juego_label(juego)}", pad=15, fontweight="bold")
    ax.set_ylabel("Cantidad de Tokens (Escala Log)", labelpad=10)
    ax.set_xlabel("Configuración del Modelo", labelpad=10)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=10)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5,-0.2), ncol=2, frameon=True)
    
    fname = f"boxplot_tokens_{juego}{suffix}.pdf"
    plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"   🪙  {fname}")

def graficar_boxplots_eficiencia(raw_dir, output_dir):
    df_time, df_token = recolectar_datos_raw(raw_dir)
    if df_time.empty or df_token.empty:
        print("⚠️  No se extrajeron datos raw válidos de los JSONL. Saltando diagramas de caja...")
        return
        
    df_time["Model_Visual"] = df_time["Model"].apply(limpiar_nombre)
    df_token["Model_Visual"] = df_token["Model"].apply(limpiar_nombre)
    juegos = df_time["Game"].unique()
    
    for juego in juegos:
        df_time_j = df_time[df_time["Game"] == juego]
        df_token_j = df_token[df_token["Game"] == juego]
        
        if juego.lower() in ["notconnectfour", "not connect 4"]:
            modelos = df_time_j["Model_Visual"].unique()
            
            modelos_aislados = [m for m in modelos if "alpha-1" in m.lower() or "checkpoint" in m.lower()]
            modelos_base = [m for m in modelos if m not in modelos_aislados]
            
            if modelos_base:
                generar_boxplot_tiempo(df_time_j[df_time_j["Model_Visual"].isin(modelos_base)], juego, output_dir, "_comparativo_base")
                generar_boxplot_tokens(df_token_j[df_token_j["Model_Visual"].isin(modelos_base)], juego, output_dir, "_comparativo_base")
                
            for m in modelos_aislados:
                safe_suffix = "_" + nombre_archivo(m.split("➔")[-1].strip().lower())
                generar_boxplot_tiempo(df_time_j[df_time_j["Model_Visual"] == m], juego, output_dir, safe_suffix)
                generar_boxplot_tokens(df_token_j[df_token_j["Model_Visual"] == m], juego, output_dir, safe_suffix)
        else:
            generar_boxplot_tiempo(df_time_j, juego, output_dir)
            generar_boxplot_tokens(df_token_j, juego, output_dir)

# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
def main(input_csv, raw_dir, output_dir):
    if not os.path.exists(input_csv):
        print(f"❌ No se encontró el CSV: {input_csv}"); return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    print(f"CSV cargado: {df.shape[0]} filas | Juegos: {df['Game'].unique()} | Modelos: {len(df['Model'].unique())}")

    configurar_estilo()

    print("\n1. Generando gráficas de barras apiladas (Desempeño en tonos pastel fríos)...")
    graficar_rendimiento_apilado(df, output_dir)

    print("\n2. Generando gráficas de tasas de error (Robustez)...")
    graficar_tasas_error(df, output_dir)

    print("\n3. Generando diagramas de caja y bigotes (Eficiencia Raw con Escala Logarítmica)...")
    graficar_boxplots_eficiencia(raw_dir, output_dir)

    print(f"\n✅ Todos los gráficos generados en el directorio: {output_dir}/")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      "-i", default="consolidated_base.csv", help="Archivo CSV consolidado")
    parser.add_argument("--raw-dir",    "-r", default="results", help="Directorio con los archivos json/jsonl")
    parser.add_argument("--output-dir", "-o", default="graficos_finales_base", help="Directorio de salida para los PDFs")
    args = parser.parse_args()
    main(args.input, args.raw_dir, args.output_dir)