#!/usr/bin/env python3
"""汇总 K 折交叉验证结果，输出 mean ± std"""
import re
import sys
import statistics as s

model = sys.argv[1] if len(sys.argv) > 1 else 'unet'

metrics = {"liver": [], "tumor": [], "combined": []}
print(f"\n{'='*55}")
print(f"  {model.upper()} — 5-Fold Cross Validation Summary")
print(f"{'='*55}")
print(f"{'Fold':<6} {'Dice_Liver':>10} {'Dice_Tumor':>10} {'Combined':>10}")
print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10}")

for f in range(5):
    try:
        with open(f"logs/{model}_fold{f}.log") as fh:
            text = fh.read()
    except FileNotFoundError:
        print(f"  {f:<4}  {'SKIPPED':>10}")
        continue

    liver = re.search(r"best_dice_liver:\s*([\d.]+)", text)
    tumor = re.search(r"best_dice_tumor:\s*([\d.]+)", text)
    comb  = re.search(r"Best combined_dice:\s*([\d.]+)", text)

    if liver and tumor and comb:
        lv, tv, cv = float(liver.group(1)), float(tumor.group(1)), float(comb.group(1))
        metrics["liver"].append(lv)
        metrics["tumor"].append(tv)
        metrics["combined"].append(cv)
        print(f"  {f:<4}  {lv:10.4f}  {tv:10.4f}  {cv:10.4f}")
    else:
        print(f"  {f:<4}  {'NOT FOUND':>10}")
        print(f"    DEBUG: liver={liver}, tumor={tumor}, comb={comb}")

print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10}")
n = len(metrics["liver"])
if n >= 2:
    mean_l = s.mean(metrics["liver"])
    mean_t = s.mean(metrics["tumor"])
    mean_c = s.mean(metrics["combined"])
    std_l  = s.stdev(metrics["liver"])
    std_t  = s.stdev(metrics["tumor"])
    std_c  = s.stdev(metrics["combined"])
    print(f"  mean   {mean_l:10.4f}  {mean_t:10.4f}  {mean_c:10.4f}")
    print(f"  ±std   {std_l:10.4f}  {std_t:10.4f}  {std_c:10.4f}")
elif n == 1:
    print(f"  mean   {metrics['liver'][0]:10.4f}  {metrics['tumor'][0]:10.4f}  {metrics['combined'][0]:10.4f}")
else:
    print("  No valid results found — check log directory")
print(f"{'='*55}\n")
