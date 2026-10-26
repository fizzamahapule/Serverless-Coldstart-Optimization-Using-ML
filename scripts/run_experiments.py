from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models import load_model
from src.features import FEATURES
from src.policy import priority_score, recommend

def main():
    df=pd.read_csv(ROOT/"data"/"cold_start_training.csv")
    model=load_model("active")
    scenarios=[]
    rng=np.random.default_rng(10)
    for scenario in ["sparse","steady","bursty","sudden_spike"]:
        for i in range(250):
            if scenario=="sparse": inv5=int(rng.integers(0,2)); inv1=int(rng.integers(0,10)); idle=float(rng.integers(900,7200)); burst=float(rng.uniform(.2,1.5))
            elif scenario=="steady": inv5=int(rng.integers(5,25)); inv1=int(rng.integers(60,300)); idle=float(rng.integers(1,300)); burst=float(rng.uniform(.7,1.5))
            elif scenario=="bursty": inv5=int(rng.integers(0,50)); inv1=int(rng.integers(10,200)); idle=float(rng.integers(1,1800)); burst=float(rng.uniform(2,8))
            else: inv5=int(rng.integers(30,150)); inv1=int(rng.integers(30,300)); idle=float(rng.integers(1,120)); burst=float(rng.uniform(3,12))
            row={"invocations_5m":inv5,"invocations_1h":inv1,"seconds_since_last":idle,"interval_cv":float(rng.uniform(.1,3)),
             "memory_mb":int(rng.choice([128,256,512,1024])),"package_mb":float(rng.uniform(3,80)),"runtime_code":0,
             "burst_ratio":burst,"previous_cold_rate":float(rng.uniform(0,.4)),"history_count":inv1}
            risk=float(model.predict_proba(pd.DataFrame([[row[f] for f in FEATURES]],columns=FEATURES))[0,1])
            for policy in ["on_demand","fixed_threshold","ml_only","adaptive_budget_aware"]:
                action="ON-DEMAND"
                if policy=="fixed_threshold" and risk>.6: action="WARM"
                elif policy=="ml_only" and risk>.65: action="WARM"
                elif policy=="adaptive_budget_aware":
                    priority=priority_score(risk,inv1,burst>=2.5,.7,.00002,.01)
                    action=recommend(risk,priority,.01,.00002,False)[0]
                scenarios.append({"scenario":scenario,"policy":policy,"risk":risk,"action":action,"invocations_5m":inv5})
    out=pd.DataFrame(scenarios)
    summary=out.groupby(["scenario","policy"]).agg(rows=("risk","size"),mean_predicted_risk=("risk","mean"),
       warm_decisions=("action",lambda x:int(x.astype(str).str.contains("WARM").sum()))).reset_index()
    path=ROOT/"artifacts"/"experiment_summary.csv"; path.parent.mkdir(exist_ok=True)
    summary.to_csv(path,index=False); print(summary.to_string(index=False)); print("\nWrote",path)
    print("\nNOTE: this is a synthetic policy simulation, not measured AWS latency/cost or a real cold-start avoidance experiment.")
if __name__=="__main__": main()
