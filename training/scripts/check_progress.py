"""查看消融实验训练进度"""
import os, sys, time, glob

LOGDIR = 'logs/ablation'
if not os.path.isdir(LOGDIR):
    print('No ablation logs found')
    sys.exit(0)

for logfile in sorted(glob.glob(f'{LOGDIR}/*.log')):
    name = os.path.basename(logfile).replace('.log', '')
    mtime = os.path.getmtime(logfile)
    age_min = (time.time() - mtime) / 60

    with open(logfile, 'r', encoding='utf-16-le', errors='replace') as f:
        lines = f.readlines()

    epochs = [l for l in lines if 'Epoch [' in l and 'Train Loss' in l]
    best = [l for l in lines if 'Best model saved' in l]

    if not epochs:
        status = 'not started' if age_min > 10 else 'starting...'
        print(f'{name:<22} {status}')
        continue

    last_epoch = epochs[-1].strip()
    n = len(epochs)
    best_info = best[-1].strip().replace('  >>> Best model saved (', '').rstrip(')') if best else 'N/A'

    # Check if running (log updated recently)
    running = '▶' if age_min < 5 else '✓' if 'Finished' in lines[-5:] else '⏸'

    print(f'{running} {name}')
    print(f'  Epochs: {n}  |  Best: {best_info}')
    print(f'  Last:   {last_epoch}')
    if age_min > 5:
        print(f'  Idle for {age_min:.0f} min')
    print()
