"""将消融实验配置的数据路径修改为本地路径"""
import yaml, glob

for f in glob.glob(r'experiments\configs\ablation_*.yaml'):
    with open(f) as fh:
        d = yaml.safe_load(fh)
    d['data_root'] = r'H:\LiTS'
    d['save_dir'] = 'models/ablation'
    with open(f, 'w') as fh:
        yaml.dump(d, fh, default_flow_style=False, allow_unicode=True)
    print(f'Fixed: {f}')
print('Done')
