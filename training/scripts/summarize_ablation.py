#!/usr/bin/env python3
"""汇总消融实验结果，与基线对比"""
import re, os

# 基线（来自服务器 fold 0, seed=3407）
BASELINE = {
    "liver": 0.8919, "tumor": 0.7666, "combined": 0.8167
}

EXPERIMENTS = [
    ("weight=1", "ablation_weight1"),
    ("weight=5", "ablation_weight5"),
    ("CE only",  "ablation_ceonly"),
    ("Dice only","ablation_diceonly"),
]

print(f"\n{'='*70}")
print(f"  Ablation Study Summary (U-Net, fold 0, seed=3407)")
print(f"{'='*70}")
print(f"{'Experiment':<16} {'Dice_Liver':>10} {'Dice_Tumor':>10} {'Combined':>10}  {'vs Baseline':>12}")
print(f"{'-'*16} {'-'*10} {'-'*10} {'-'*10}  {'-'*12}")
print(f"{'BASELINE':<16} {BASELINE['liver']:10.4f} {BASELINE['tumor']:10.4f} {BASELINE['combined']:10.4f}")
print()

for name, exp_id in EXPERIMENTS:
    logfile = f"logs/ablation/{exp_id}.log"
    if not os.path.exists(logfile):
        print(f"  {name:<14}  NOT FOUND")
        continue
    with open(logfile) as f:
        text = f.read()
    liver = re.search(r"best_dice_liver:\s*([\d.]+)", text)
    tumor = re.search(r"best_dice_tumor:\s*([\d.]+)", text)
    comb  = re.search(r"Best combined_dice:\s*([\d.]+)", text)
    if liver and tumor and comb:
        lv = float(liver.group(1))
        tv = float(tumor.group(1))
        cv = float(comb.group(1))
        dl = lv - BASELINE["liver"]
        dt = tv - BASELINE["tumor"]
        print(f"  {name:<14}  {lv:10.4f}  {tv:10.4f}  {cv:10.4f}  {'▲' if dl>0 else '▼'}{abs(dl):.4f}/{'▲' if dt>0 else '▼'}{abs(dt):.4f} L/T")

print(f"{'='*70}\n")
