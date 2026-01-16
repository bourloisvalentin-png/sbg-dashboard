# app_sbg_base.py

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np 
import io 

# =========================
# CONFIG GÉNÉRALE PAGE
# =========================
st.set_page_config(
    page_title="SBG - Tableau de bord",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CONSTANTES / CHEMINS
# =========================
FICHIER_MASTER = "BASE_TRAVAIL.xlsx"

# Colonnes d'identification à exclure
COL_IDENTIFICATION = [
    'inputId', 'numReponse', 'programme', 'date', 'Statuts'  # ✅ Ajout Statuts
]

# =========================
# FONCTIONS
# =========================
@st.cache_data
def charger_base(filename: str) -> pd.DataFrame:
    """Charge Excel depuis repo GitHub."""
    try:
        return pd.read_excel(filename)
    except FileNotFoundError:
        st.error(f"❌ Fichier `{filename}` manquant dans repo GitHub")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lecture Excel : {e}")
        return pd.DataFrame()

def detecter_colonne_programme(df: pd.DataFrame) -> str | None:
    """Détecte la colonne programme."""
    return 'programme' if 'programme' in df.columns else None

def detecter_colonne_statut(df: pd.DataFrame) -> str | None:
    """Détecte colonne Statuts (flexible)."""
    for col in ['Statuts', 'statut', 'Statut', 'STATUS', 'etat', 'Etat']:
        if col in df.columns:
            return col
    return None

def colonnes_analyse_disponibles(df: pd.DataFrame) -> list[str]:
    """Colonnes numériques pour analyse."""
    cols_num = df.select_dtypes(include=['number']).columns.tolist()
    cols_analyse = [col for col in cols_num if col.lower() not in 
                   [c.lower() for c in COL_IDENTIFICATION]]
    return cols_analyse

def toggle_apercu(df_filtre: pd.DataFrame, selected_cols: list, afficher: bool) -> None:
    """Affiche/masque l'aperçu."""
    if not afficher:
        return
        
    st.subheader("👀 Aperçu des données")
    
    if selected_cols:
        cols_affichage = [col for col in COL_IDENTIFICATION + selected_cols if col in df_filtre.columns]
        st.success("✅ Données prêtes pour analyse")
        st.dataframe(df_filtre[cols_affichage].head(20), use_container_width=True)
    else:
        cols_id_only = [col for col in COL_IDENTIFICATION if col in df_filtre.columns]
        st.info("⏳ Sélectionnez des colonnes d'analyse")
        if cols_id_only:
            st.dataframe(df_filtre[cols_id_only].head(20), use_container_width=True)

# =========================
# EN-TÊTE
# =========================
st.title("📊 SBG - Tableau de bord")

with st.spinner("Chargement des données..."):
    df = charger_base(FICHIER_MASTER)

if df.empty:
    st.error("❌ Impossible de charger `BASE_TRAVAIL.xlsx`.")
    st.stop()

st.success(f"✅ Base chargée : {len(df)} lignes, {len(df.columns)} colonnes.")

COL_PROG = detecter_colonne_programme(df)
COL_STATUT = detecter_colonne_statut(df)

if COL_PROG is None:
    st.error("❌ Colonne 'programme' non trouvée.")
    st.stop()

colonnes_analyse = colonnes_analyse_disponibles(df)

# =========================
# BARRE LATÉRALE
# =========================
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.markdown("---")
    
    # ✅ STATUT (NOUVEAU - AVANT Programme)
    st.header("📋 Statut")
    if COL_STATUT and COL_STATUT in df.columns:
        statuts_dispo = sorted(df[COL_STATUT].dropna().unique())
        selected_statuts = st.multiselect(
            "Sélectionnez statut(s)",
            options=statuts_dispo,
            help="Filtre par statut"
        )
        st.caption(f"({len(statuts_dispo)} disponibles)")
    else:
        selected_statuts = []
        st.warning("⚠️ Colonne Statuts non trouvée")
    
    st.markdown("---")
    
    # Programme
    st.header("📋 Programme")
    programmes_dispo = sorted(df[COL_PROG].dropna().unique())
    selected_progs = st.multiselect(
        "Sélectionnez programme(s)",
        options=programmes_dispo,
        help="Choisissez un ou plusieurs programmes"
    )
    
    st.markdown("---")
    
    # Colonnes
    st.header("📊 Colonnes d'analyse")
    if colonnes_analyse:
        selected_cols = st.multiselect(
            "Colonnes à analyser",
            options=colonnes_analyse
        )
        st.caption(f"({len(colonnes_analyse)} disponibles)")
    else:
        selected_cols = []
        st.warning("⚠️ Aucune colonne numérique")
    
    st.markdown("---")
    
    # Toggle
    st.header("👁️ Affichage")
    afficher_apercu = st.checkbox("Afficher l'aperçu", value=True)
    afficher_global = st.checkbox("Afficher analyses globales", value=True)

# =========================
# FILTRES SÉQUENTIELS ✅ NOUVEAU
# =========================
df_filtre = df.copy()

# 1. Statut d'abord
if selected_statuts and COL_STATUT:
    df_filtre = df_filtre[df_filtre[COL_STATUT].isin(selected_statuts)]

# 2. Programme ensuite  
if selected_progs:
    df_filtre = df_filtre[df_filtre[COL_PROG].isin(selected_progs)]

nb_individus = df_filtre['inputId'].nunique() if 'inputId' in df_filtre.columns else len(df_filtre)

# =========================
# PRINCIPALE
# =========================
st.header("🔍 Données")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total", len(df))
with col2: st.metric("Filtré", len(df_filtre))
with col3: st.metric("Colonnes", len(selected_cols))
with col4: st.metric("Individus", nb_individus)

# Infos filtres
infos_filtre = []
if selected_statuts: infos_filtre.append(f"{len(selected_statuts)} statut(s)")
if selected_progs: infos_filtre.append(f"{len(selected_progs)} prog")
if selected_cols: infos_filtre.append(f"{len(selected_cols)} cols")
if infos_filtre:
    st.info(f"✅ Filtres : {' • '.join(infos_filtre)} → {nb_individus} ind")
else:
    st.info("ℹ️ Aucun filtre")

toggle_apercu(df_filtre, selected_cols, afficher_apercu)

# =========================
# ANALYSES GLOBALES
# =========================
if selected_cols and len(df_filtre) > 0 and afficher_global:
    st.markdown("---")
    
    # Histogrammes
    st.subheader("📊 Distributions")
    cols_par_ligne = 3
    for i in range(0, len(selected_cols), cols_par_ligne):
        cols_chunk = selected_cols[i:i+cols_par_ligne]
        cols_layout = st.columns(cols_par_ligne)
        for j, col in enumerate(cols_chunk):
            with cols_layout[j]:
                fig = px.histogram(df_filtre, x=col, title=col, height=300)
                st.plotly_chart(fig, use_container_width=True)

# =========================
# TESTS T1/T2
# =========================
st.markdown("---")
st.subheader("🧪 Tests T1 vs T2")

col_t = next((c for c in ['numReponse', 'num_reponse', 'T'] if c in df_filtre.columns), None)

if col_t and 1 in df_filtre[col_t].values and 2 in df_filtre[col_t].values:
    # Sidebar tests
    with st.sidebar:
        st.markdown("---")
        st.header("🔬 Tests T1/T2")
        selected_test = st.radio("Test", ["Auto", "t-test", "Wilcoxon"], key="test")
        show_dist = st.checkbox("Distributions", value=True, key="dist")
    
    for col in selected_cols:
        df_wide = (df_filtre[df_filtre[col_t].isin([1,2])]
                  .groupby(['inputId', col_t])[col].first().unstack()
                  .rename(columns={1: 'T1', 2: 'T2'})
                  .dropna(subset=['T1', 'T2'])
                  .assign(delta=lambda x: x['T2'] - x['T1'])
                  .reset_index())
        
        if df_wide.empty: 
            st.warning(f"⚠️ Pas de paires T1/T2 pour {col}")
            continue
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("T1", f"{df_wide['T1'].mean():.1f}")
        with col2: st.metric("T2", f"{df_wide['T2'].mean():.1f}")
        with col3: st.metric("n", len(df_wide))
        
        # 🔥 TESTS CALCULÉS ICI (côte-à-côte)
        shapiro_p = stats.shapiro(df_wide['delta'])[1]
        normalite = shapiro_p > 0.05
        
        if selected_test == "Auto":
            if normalite:
                result = stats.ttest_rel(df_wide['T2'], df_wide['T1'])
                p_val, stat_val = result.pvalue, result.statistic
                test_name = "t-test"
            else:
                res = stats.wilcoxon(df_wide['T1'], df_wide['T2'])
                p_val, stat_val = res.pvalue, res.statistic
                test_name = "Wilcoxon"
        elif selected_test == "t-test apparié":
            result = stats.ttest_rel(df_wide['T2'], df_wide['T1'])
            p_val, stat_val = result.pvalue, result.statistic
            test_name = "t-test"
        else:
            res = stats.wilcoxon(df_wide['T1'], df_wide['T2'])
            p_val, stat_val = res.pvalue, res.statistic
            test_name = "Wilcoxon"
        
        # Test stat
        with col4:
            st.metric(test_name, f"{stat_val:.2f}")
        
        # p-value
        with col5:
            emoji_p = "✅" if p_val < 0.05 else "❌"
            st.metric("p", f"{p_val:.3f} {emoji_p}", delta=None)
                
        # 🔥 GRAPHIQUE PRINCIPAL : Boxplot + points
        fig_box_main = px.box(
            df_wide.melt(id_vars=['inputId'], value_vars=['T1', 'T2'], 
                        var_name='Temps', value_name=col),
            x='Temps', y=col, points="all",
            color='Temps', boxmode='group',
            title=f"{col} : T1 vs T2 (n={n_paires})",
            height=450
        )
        fig_box_main.update_traces(pointpos=0.3)
        st.plotly_chart(fig_box_main, use_container_width=True)
        
        # 🔥 RAPPORT SCIENTIFIQUE CLASSIQUE
        st.markdown("---")
        st.caption("**📄 Rapport publication**")
        
        et_t1 = df_wide['T1'].std() / np.sqrt(n_paires)
        et_t2 = df_wide['T2'].std() / np.sqrt(n_paires)
        cohens_d = abs(df_wide['delta'].mean()) / df_wide['delta'].std()
        
        df_stat = n_paires - 1 if 't-test' in test_name else ""
        
        rapport = f"""
**{col}** : T1 (M = {df_wide['T1'].mean():.1f}, ET = {et_t1:.1f}) → 
T2 (M = {df_wide['T2'].mean():.1f}, ET = {et_t2:.1f})

**{test_name}**({df_stat}) = {stat_val:.2f}, **p = {p_val:.3f}**, **d = {cohens_d:.2f}**
        """
        
        st.code(rapport, language="markdown")
        
        # Interprétation
        if p_val < 0.05:
            effet = "significative" if cohens_d < 0.5 else "forte"
            st.success(f"✅ **Amélioration {effet}** (d = {cohens_d:.2f})")
        else:
            st.info(f"ℹ️ **Pas de différence significative** (p = {p_val:.3f})")
        
        # Distributions optionnelles
        if show_dist:
            st.markdown("**📊 Distributions détaillées**")
            fig_dist = make_subplots(rows=1, cols=2, subplot_titles=(f'{col} T1', f'{col} T2'))
            fig_dist.add_trace(go.Histogram(x=df_wide['T1'], opacity=0.7, marker_color='blue', nbinsx=15), 1, 1)
            fig_dist.add_trace(go.Histogram(x=df_wide['T2'], opacity=0.7, marker_color='orange', nbinsx=15), 1, 2)
            fig_dist.add_vline(df_wide['T1'].mean(), line_dash="dash", line_color="blue", row=1, col=1)
            fig_dist.add_vline(df_wide['T2'].mean(), line_dash="dash", line_color="orange", row=1, col=2)
            fig_dist.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("---")

else:

    st.error(f"❌ Pas de paires T1/T2 dans `{col_t}`")


