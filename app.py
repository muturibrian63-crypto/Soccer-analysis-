import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "api_football_daily.py"
OUTPUT = ROOT / "output"

LEAGUES = {
    "England — Premier League": "E0",
    "England — Championship": "E1",
    "England — League One": "E2",
    "England — League Two": "E3",
    "England — National League": "EC",
    "Scotland — Premiership": "SC0",
    "Scotland — Championship": "SC1",
    "Scotland — League One": "SC2",
    "Scotland — League Two": "SC3",
    "Germany — Bundesliga": "D1",
    "Germany — 2. Bundesliga": "D2",
    "Italy — Serie A": "I1",
    "Italy — Serie B": "I2",
    "Spain — La Liga": "SP1",
    "Spain — La Liga 2": "SP2",
    "France — Ligue 1": "F1",
    "France — Ligue 2": "F2",
    "Netherlands — Eredivisie": "N1",
    "Belgium — Pro League": "B1",
    "Portugal — Primeira Liga": "P1",
    "Turkey — Super Lig": "T1",
    "Greece — Super League": "G1",
}
DEFAULT_LEAGUES = [
    "England — Premier League", "Germany — Bundesliga", "Italy — Serie A",
    "Spain — La Liga", "France — Ligue 1", "Netherlands — Eredivisie",
]

st.set_page_config(page_title="Soccer Value Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Soccer Value Predictor")
st.caption("Free-data football analysis • model estimates, not guaranteed betting tips")

with st.sidebar:
    st.header("Analysis settings")
    date_choice = st.selectbox("Fixture date", ["Today", "Tomorrow", "Choose a date"])
    if date_choice == "Today":
        target_date = date.today()
    elif date_choice == "Tomorrow":
        target_date = date.today() + timedelta(days=1)
    else:
        target_date = st.date_input("Select date", value=date.today())

    selected_names = st.multiselect(
        "Leagues to analyse", list(LEAGUES.keys()), default=DEFAULT_LEAGUES,
        help="The free source covers selected leagues only, not every country or competition.",
    )
    seasons = st.slider("Historical seasons", min_value=2, max_value=8, value=3)
    min_edge = st.slider("Minimum probability edge", 0.0, 0.20, 0.04, 0.01, format="%.0f%%")
    min_ev = st.slider("Minimum estimated expected value", 0.0, 0.25, 0.03, 0.01, format="%.0f%%")
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

st.info("Markets shown: 1X2, double chance, Over/Under 2.5, BTTS and most-likely correct score. Value screening is possible only where the free data source provides odds.")

if run_button:
    if not selected_names:
        st.error("Choose at least one league before running the analysis.")
        st.stop()

    selected_codes = ",".join(LEAGUES[name] for name in selected_names)
    OUTPUT.mkdir(exist_ok=True)
    command = [
        sys.executable, str(SCRIPT), "--date", target_date.isoformat(),
        "--outdir", str(OUTPUT), "--leagues", selected_codes,
        "--seasons", str(seasons), "--min-edge", str(min_edge), "--min-ev", str(min_ev),
    ]
    status = st.status("Downloading match data and running the models… this may take a few minutes.", expanded=True)
    try:
        result = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True, timeout=900
        )
        if result.stdout:
            status.write(result.stdout[-12000:])
        if result.returncode != 0:
            status.update(label="Analysis did not finish", state="error")
            st.error("The analysis script returned an error. See details below.")
            if result.stderr:
                st.code(result.stderr[-12000:])
            st.stop()
        json_path = OUTPUT / "predictions.json"
        if not json_path.exists():
            status.update(label="No report was produced", state="error")
            st.error("The script finished but did not create predictions.json.")
            st.stop()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        status.update(label="Analysis complete", state="complete", expanded=False)
        predictions = payload.get("predictions", [])
        backtest = payload.get("backtest", {})
        st.session_state["soccer_payload"] = payload
        st.session_state["soccer_date"] = str(target_date)
        st.success(f"Finished. Found {len(predictions)} fixture(s) in the selected source data.")
    except subprocess.TimeoutExpired:
        status.update(label="Analysis timed out", state="error")
        st.error("The run took over 15 minutes. Try fewer leagues or fewer historical seasons.")
    except Exception as exc:
        status.update(label="Analysis failed", state="error")
        st.exception(exc)

payload = st.session_state.get("soccer_payload")
if payload:
    predictions = payload.get("predictions", [])
    backtest = payload.get("backtest", {})
    st.subheader(f"Predictions — {st.session_state.get('soccer_date', '')}")
    if backtest.get("matches"):
        a, b, c, d = st.columns(4)
        a.metric("Holdout matches", backtest.get("matches", 0))
        a.metric("1X2 accuracy", f"{backtest.get('accuracy', 0):.1%}")
        b.metric("Brier score", f"{backtest.get('mean_brier_1x2', 0):.3f}")
        c.metric("Log loss", f"{backtest.get('mean_log_loss', 0):.3f}")
        d.metric("Historical test ROI", "N/A" if backtest.get("value_bet_roi") is None else f"{backtest['value_bet_roi']:.1%}")
        st.caption("Historical holdout metrics are diagnostic only; they do not prove future profitability.")

    if not predictions:
        st.warning("No supported fixtures were found for this date. Try another date or select more leagues.")
    else:
        rows = []
        for p in predictions:
            rows.append({
                "Kick-off (EAT)": p.get("kickoff_nairobi"),
                "League": p.get("league"),
                "Fixture": f"{p.get('home')} vs {p.get('away')}",
                "Home %": round(100 * p["model_1x2"]["home_win"], 1),
                "Draw %": round(100 * p["model_1x2"]["draw"], 1),
                "Away %": round(100 * p["model_1x2"]["away_win"], 1),
                "Over 2.5 %": round(100 * p["goal_model"]["over25"], 1),
                "Under 2.5 %": round(100 * p["goal_model"]["under25"], 1),
                "BTTS Yes %": round(100 * p["goal_model"]["btts_yes"], 1),
                "BTTS No %": round(100 * p["goal_model"]["btts_no"], 1),
                "Double chance 1X %": round(100 * p["double_chance"]["1X"], 1),
                "Double chance X2 %": round(100 * p["double_chance"]["X2"], 1),
                "Double chance 12 %": round(100 * p["double_chance"]["12"], 1),
                "Most likely score": p.get("correct_score_most_likely"),
                "Scoreline probability %": round(100 * p.get("correct_score_probability", 0), 1),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions as CSV", df.to_csv(index=False).encode("utf-8"),
            file_name=f"soccer_predictions_{st.session_state.get('soccer_date', 'report')}.csv",
            mime="text/csv",
        )

        st.subheader("Fixture details and value candidates")
        for p in predictions:
            with st.expander(f"{p['home']} vs {p['away']} — {p['league']}"):
                st.write(f"**Expected goals:** {p['expected_goals']['home']:.2f} – {p['expected_goals']['away']:.2f}")
                st.write(f"**1X2:** Home {p['model_1x2']['home_win']:.1%} · Draw {p['model_1x2']['draw']:.1%} · Away {p['model_1x2']['away_win']:.1%}")
                st.write(f"**Double chance:** 1X {p['double_chance']['1X']:.1%} · X2 {p['double_chance']['X2']:.1%} · 12 {p['double_chance']['12']:.1%}")
                st.write(f"**Goals:** Over 2.5 {p['goal_model']['over25']:.1%} · Under 2.5 {p['goal_model']['under25']:.1%}")
                st.write(f"**BTTS:** Yes {p['goal_model']['btts_yes']:.1%} · No {p['goal_model']['btts_no']:.1%}")
                st.write(f"**Most likely correct score:** {p['correct_score_most_likely']} ({p['correct_score_probability']:.1%} model probability)")
                candidates = p.get("potential_value_bets", [])
                if candidates:
                    st.dataframe(pd.DataFrame([{
                        "Market": x["market"], "Model probability": f"{x['probability']:.1%}",
                        "Odds": x["odds"], "Edge": f"{x['edge']:.1%}",
                        "Estimated EV": f"{x['expected_value']:+.1%}",
                    } for x in candidates]), use_container_width=True, hide_index=True)
                else:
                    st.write("No candidate passed the selected value thresholds using available odds.")

st.divider()
st.caption("Data source: football-data.co.uk. Coverage is limited to selected leagues and available fixtures/odds. Predictions are uncertain estimates, not guaranteed tips. Never bet money you cannot afford to lose.")
