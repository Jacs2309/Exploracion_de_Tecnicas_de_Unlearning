import argparse
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, LogLocator
from collections import defaultdict
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN VISUAL (PALETAS COMPLETAMENTE DIFERENTES)
# ──────────────────────────────────────────────────────────────

# Gráfica 1: Desempeño Competitivo (Verde Menta, Arena, Terracota)
COLORES_METRICA = {
    "Win":  "#2A9D8F",  
    "Draw": "#E9C46A",  
    "Loss": "#E76F51",  
}

# Gráfica 2: Validez de Jugadas (Azul Acero, Celeste, Azul Marino)
COLORES_ERRORES = {
    "Jugadas Ilegales": "#457B9D",  
    "Error de Formato": "#A8DADC",  
    "Tasa de Forfeit":  "#1D3557",  
}

# Gráfica 3: Costos de Jugada (Verde Oliva, Piedra, Salvia Oscuro)
COLORES_MODELOS_G3 = {
    "Base": "#84A98C", 
    "TVN":  "#CAD2C5", 
    "NPO":  "#52796F"  
}

FAMILIA_COLOR = {
    "suicide":        "#F7F9FC",
    "notconnectfour": "#F4F9F6",
    "tictactoe":      "#F7F9FC",
    "connectfour":    "#F4F9F6",
}

ORDEN_OPONENTES = ["random", "legal", "greedy", "mcs"]
ORDEN_METRICAS  = ["Win", "Draw", "Loss"]

def configurar_estilo():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times New Roman", "Palatino", "serif"],
        "font.size": 22,             # Textos generales más grandes
        "axes.labelsize": 28,        # Etiquetas de ejes aumentadas
        "axes.titlesize": 30,        # Títulos más notorios
        "xtick.labelsize": 24,       # Textos en X (oponentes y modelos) aumentados
        "ytick.labelsize": 20,       # Números en Y aumentados
        "axes.labelweight": "bold",  # Etiquetas siempre en negrita
        "axes.titleweight": "bold",  # Títulos siempre en negrita
        "figure.dpi": 300,           
        "savefig.dpi": 300,
        "axes.edgecolor": "#555555", 
        "axes.linewidth": 1.2,
        "grid.color": "#e6e9ec", 
        "grid.linewidth": 0.8, 
        "grid.alpha": 0.8,
    })

def limpiar_nombre_corto(nombre):
    """Filtra y extrae solo NPO, TVN o Base."""
    if not isinstance(nombre, str): return "Base"
    
    n_lower = nombre.lower()
    # Prioridad: NPO
    if "npo" in n_lower:
        return "NPO"
    # Prioridad: TVN (incluye alpha, beta, tvn, o nombres específicos de suicide)
    elif any(keyword in n_lower for keyword in ["alpha", "beta", "tvn", "suicide-alpha"]):
        return "TVN"
    # Todo lo demás es Base
    return "Base"

def nombre_archivo(nombre):
    return nombre.replace("/", "_").replace("\\", "_").replace("|", "_").replace(":", "_").replace(" ", "_")

def fondo_juego(ax, juego):
    ax.set_facecolor(FAMILIA_COLOR.get(juego.lower(), "#Fcfcfc"))

def juego_label(juego):
    return juego.replace("notconnectfour","Not Connect 4").replace("suicide","Suicide TTT").replace("tictactoe","Tic-Tac-Toe").replace("connectfour","Connect 4").upper()

# ──────────────────────────────────────────────────────────────
# GRÁFICO 1: DESEMPEÑO COMPETITIVO
# ──────────────────────────────────────────────────────────────
def graficar_rendimiento_consolidado(df, output_dir):
    for juego in df["Game"].unique():
        df_j = df[df["Game"]==juego].copy()
        if df_j.empty: continue

        df_j["Model_Clean"] = df_j["Model"].apply(limpiar_nombre_corto)
        orden = ["Base", "TVN", "NPO"]
        modelos = [m for m in orden if m in df_j["Model_Clean"].unique()]
        n_mod = len(modelos)

        df_j["Opponent"] = pd.Categorical(df_j["Opponent"], categories=ORDEN_OPONENTES, ordered=True)
        oponentes = sorted(df_j["Opponent"].dropna().unique())
        if not oponentes: continue

        fig_width = max(14, len(oponentes) * n_mod * 1.8) # Más ancho para que entren los textos horizontales
        fig, ax = plt.subplots(figsize=(fig_width, 8.5))
        fondo_juego(ax, juego)

        x = np.arange(len(oponentes))
        width = 0.8 / n_mod
        offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_mod)

        handles_met, labels_met = [], []
        metricas_vistas = set()

        for i_op, op in enumerate(oponentes):
            for i_mod, modelo in enumerate(modelos):
                df_sub = df_j[(df_j["Model_Clean"]==modelo) & (df_j["Opponent"]==op)]
                pos = x[i_op] + offsets[i_mod]

                if df_sub.empty: continue

                win = df_sub["Win_Rate"].mean() if "Win_Rate" in df_sub.columns else 0
                draw = df_sub["Draw_Rate"].mean() if "Draw_Rate" in df_sub.columns else 0
                loss = df_sub["Loss_Rate"].mean() if "Loss_Rate" in df_sub.columns else 0
                
                vals = [win, draw, loss]
                bottom = 0

                for metrica, val in zip(ORDEN_METRICAS, vals):
                    if val > 0:
                        bar = ax.bar(pos, val, width, bottom=bottom,
                                     color=COLORES_METRICA[metrica],
                                     edgecolor="#555555", linewidth=0.7)
                        
                        if val > 0.07: 
                            ax.text(pos, bottom + val/2, f"{val:.2f}",
                                    ha="center", va="center", fontsize=14, color="#222222", fontweight="bold")
                        bottom += val

                        if metrica not in metricas_vistas:
                            metricas_vistas.add(metrica)
                            handles_met.append(bar[0])
                            labels_met.append(metrica)

                # NOMBRES HORIZONTALES: Aumenté el tamaño y quité la rotación
                ax.text(pos, -0.04, modelo, ha="center", va="top", rotation=0,
                        fontsize=16, fontweight="bold", color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([op.capitalize() for op in oponentes], fontsize=18, fontweight="bold")
        ax.tick_params(axis='x', pad=65) # Empuja las etiquetas de Oponente más hacia abajo
        
        ax.set_ylabel("Proporción de Resultados (0.0 a 1.0)", labelpad=15)
        ax.set_xlabel("Oponente GGP", labelpad=15)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.6)
        
        ax.set_title(f"Desempeño Competitivo - {juego_label(juego)}", pad=20, fontweight="bold")

        ax.legend(handles_met, labels_met, title="Resultado", loc="upper center",
                  bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc")

        fname = f"rendimiento_comparativo_unificado_{juego}.pdf"
        plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close()
        print(f"   📊 {fname}")

# ──────────────────────────────────────────────────────────────
# GRÁFICO 2: VALIDEZ DE JUGADAS (BARRAS HORIZONTALES)
# ──────────────────────────────────────────────────────────────
def graficar_tasas_error(df, output_dir):
    METRICAS_ERR = {
        "Jugadas Ilegales": ("Illegal_Rate",      COLORES_ERRORES["Jugadas Ilegales"]), 
        "Error de Formato": ("Format_Error_Rate", COLORES_ERRORES["Error de Formato"]), 
        "Tasa de Forfeit":  ("Forfeit_Rate",      COLORES_ERRORES["Tasa de Forfeit"]),
    }
    
    for juego in df["Game"].unique():
        df_j = df[df["Game"]==juego].copy()
        if df_j.empty: continue
        
        df_j["Model_Clean"] = df_j["Model"].apply(limpiar_nombre_corto)
        orden = ["Base", "TVN", "NPO"]
        modelos = [m for m in orden if m in df_j["Model_Clean"].unique()]
        n_mod = len(modelos)
        
        n_met = len(METRICAS_ERR)
        height = 0.25 
        y = np.arange(n_mod)
        offsets = np.linspace(-(n_met-1)/2*height, (n_met-1)/2*height, n_met)

        fig_height = max(8, n_mod * 3) # Aumentado
        fig, ax = plt.subplots(figsize=(14, fig_height)) # Aumentado
        fondo_juego(ax, juego)

        max_w = 0
        for i, (label, (col, color)) in enumerate(METRICAS_ERR.items()):
            vals = [df_j[df_j["Model_Clean"]==m][col].mean() if col in df_j.columns else 0 for m in modelos]
            if vals: max_w = max(max_w, max(vals))
            
            bars = ax.barh(y + offsets[i], vals, height, label=label,
                           color=color, edgecolor="#555555", linewidth=0.7, alpha=0.95)
            
            for bar in bars:
                w = bar.get_width()
                if w > 0.005:
                    offset_texto = max(0.005, max_w * 0.015)
                    ax.text(w + offset_texto, bar.get_y() + bar.get_height()/2,
                            f"{w:.2f}", ha="left", va="center", fontsize=15, fontweight="bold", color="#222222")

        ax.set_yticks(y)
        ax.set_yticklabels(modelos, fontsize=18, fontweight="bold")
        ax.invert_yaxis()
        
        ax.set_xlabel("Tasa Promedio (0.0 a 1.0)", labelpad=15)
        ax.set_title(f"Validez de Jugadas - {juego_label(juego)}", pad=20, fontweight="bold")
        
        ax.set_xlim(0, max(0.3, max_w * 1.15))
        ax.grid(axis="x", which="major", linestyle="--", alpha=0.6)
        ax.grid(visible=False,axis="y") 
        
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc")

        fname = f"errores_unificados_{juego}.pdf"
        plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close()
        print(f"   🛠️  {fname}")

# ──────────────────────────────────────────────────────────────
# GRÁFICO 3: COSTOS DE JUGADA (BARRAS AGRUPADAS MODERNAS)
# ──────────────────────────────────────────────────────────────
def recolectar_datos_raw(raw_dir):
    time_data, token_data = [], []
    baselines = ["random", "legal", "greedy", "mcs"]
    
    exp_files = glob.glob(os.path.join(raw_dir, "experiment_*.json"))
    experimentos_validos = {} 
    
    for ef in exp_files:
        try:
            with open(ef, 'r', encoding='utf-8') as f:
                data = json.load(f)
                exp = data.get("experiment", {})
                game = exp.get("game", "unknown").lower()
                p1, p2 = exp.get("p1", ""), exp.get("p2", "")
                
                results = data.get("results", [])
                if not results: continue
                experimentos_validos[(game, p1, p2)] = len(results)
                
                for r in results:
                    role_llm, llm_name = None, None
                    if game in ["connectfour", "notconnectfour"]:
                        if r.get("red_player") not in baselines: role_llm, llm_name = "red", r.get("red_player")
                        elif r.get("black_player") not in baselines: role_llm, llm_name = "black", r.get("black_player")
                    else:
                        if r.get("x_player") not in baselines: role_llm, llm_name = "x", r.get("x_player")
                        elif r.get("o_player") not in baselines: role_llm, llm_name = "o", r.get("o_player")
                            
                    if role_llm and llm_name:
                        tokens = r.get("tokens", {}).get(role_llm, {})
                        token_data.append({
                            "Game": game, "Model": llm_name,
                            "Prompt_Tokens": tokens.get("prompt", 0), "Completion_Tokens": tokens.get("completion", 0)
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
                    key = (match.get("game", "unknown").lower(), match.get("player1", ""), match.get("player2", ""))
                    partidas_candidatas[key].append({"id_match": match.get("id_match"), "timestamp": match.get("timestamp", "")})
        except: pass

    valid_match_ids = set()
    for key, matches_list in partidas_candidatas.items():
        game, p1, p2 = key
        cantidad = experimentos_validos.get((game, p1, p2), 0) or experimentos_validos.get((game, p2, p1), 0)
        if cantidad > 0:
            matches_list.sort(key=lambda x: x["timestamp"], reverse=True) 
            for item in matches_list[:cantidad]: valid_match_ids.add(item["id_match"])

    move_files = glob.glob(os.path.join(raw_dir, "moves_*.jsonl"))
    for mf in move_files:
        try:
            with open(mf, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    if d.get("id_match") not in valid_match_ids: continue
                    model = d.get("model", "unknown")
                    if model not in baselines:
                        time_data.append({"Game": d.get("game", "unknown"), "Model": model, "Execution_Time": d.get("execution_time", 0.0)})
        except: pass

    return pd.DataFrame(time_data), pd.DataFrame(token_data)

def graficar_costos_eficiencia(raw_dir, output_dir):
    df_time, df_token = recolectar_datos_raw(raw_dir)
    if df_time.empty or df_token.empty: return
        
    df_time["Model_Clean"] = df_time["Model"].apply(limpiar_nombre_corto)
    df_token["Model_Clean"] = df_token["Model"].apply(limpiar_nombre_corto)
    
    for juego in df_time["Game"].unique():
        df_time_j = df_time[df_time["Game"] == juego].copy()
        df_token_j = df_token[df_token["Game"] == juego].copy()
        
        df_t = df_time_j[["Model_Clean", "Execution_Time"]].rename(columns={"Execution_Time": "Valor"})
        df_t["Métrica"] = "Tiempo Inferencia (s)"
        
        df_pt = df_token_j[["Model_Clean", "Prompt_Tokens"]].rename(columns={"Prompt_Tokens": "Valor"})
        df_pt["Métrica"] = "Tokens Contexto"
        
        df_ct = df_token_j[["Model_Clean", "Completion_Tokens"]].rename(columns={"Completion_Tokens": "Valor"})
        df_ct["Métrica"] = "Tokens Generados"
        
        df_all = pd.concat([df_t, df_pt, df_ct], ignore_index=True)
        
        # FORZADO: Rellenar nulos para evitar que Seaborn descarte la barra de TVN
        df_all["Valor"] = df_all["Valor"].fillna(0.001) 
        
        orden = ["Base", "TVN", "NPO"]
        
        # Asegurar que Seaborn reciba datos perfectamente balanceados, aunque sean 0.001
        for met in ["Tiempo Inferencia (s)", "Tokens Contexto", "Tokens Generados"]:
            for mod in orden:
                if not ((df_all['Métrica'] == met) & (df_all['Model_Clean'] == mod)).any():
                     # Insertar fila artificial si TVN (o cualquier otro) no generó datos para esta métrica
                     df_all = pd.concat([df_all, pd.DataFrame({"Model_Clean": [mod], "Valor": [0.001], "Métrica": [met]})], ignore_index=True)
        
        orden_presentes = [m for m in orden if m in df_all["Model_Clean"].unique()]
        
        fig, ax = plt.subplots(figsize=(14, 8.5)) # Aumentado
        fondo_juego(ax, juego)
        
        sns.barplot(data=df_all, x="Métrica", y="Valor", hue="Model_Clean", 
                    hue_order=orden_presentes,
                    palette=COLORES_MODELOS_G3, 
                    ax=ax, capsize=0.08, edgecolor="#555555", linewidth=1.2)
        
        ax.set_yscale("log")
        
        loc_major = LogLocator(base=10.0, numticks=15)
        loc_minor = LogLocator(base=10.0, subs=(2.0, 5.0), numticks=15) 
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_locator(loc_major)
        ax.yaxis.set_minor_locator(loc_minor)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(formatter)
        
        ax.set_title(f"Costos de Jugada - {juego_label(juego)}", pad=20, fontweight="bold")
        ax.set_ylabel("Valor Promedio (Escala Logarítmica)", labelpad=12)
        ax.set_xlabel("", labelpad=12)
        
        plt.setp(ax.get_xticklabels(), fontsize=20, fontweight="bold")
        
        ax.legend(title="Modelo Evaluado", loc="upper center", bbox_to_anchor=(0.5, -0.1), 
                  ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc")
        
        fname = f"costos_unificados_{juego}.pdf"
        plt.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close()
        print(f"   📦 {fname}")

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

    print("\n1. Generando gráficos de desempeño competitivo...")
    graficar_rendimiento_consolidado(df, output_dir)

    print("\n2. Generando gráficos de validez de jugadas (Errores)...")
    graficar_tasas_error(df, output_dir)

    print("\n3. Generando gráficos de costos de jugada (Barras Consolidadas)...")
    graficar_costos_eficiencia(raw_dir, output_dir)

    print(f"\n✅ Todos los gráficos generados en el directorio: {output_dir}/")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      "-i", default="consolidated_con_base.csv", help="Archivo CSV consolidado")
    parser.add_argument("--raw-dir",    "-r", default="results_con_base", help="Directorio con los archivos json/jsonl")
    parser.add_argument("--output-dir", "-o", default="graficos_finales_tesis_con_base", help="Directorio de salida para los PDFs")
    args = parser.parse_args()
    main(args.input, args.raw_dir, args.output_dir)