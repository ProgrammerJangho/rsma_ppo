"""Convert existing oracle labels without changing states, splits, or objectives."""
from pathlib import Path
import sys, argparse, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rsma_aoi.utils.config import load_run_configs
from rsma_aoi.evaluation.oracle import config_hash
from rsma_aoi.envs.rsma_phy import split_payload_batch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--source-config',required=True)
    ap.add_argument('--target-config',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--limit',type=int,default=0,help='Optional per-split prefix for smoke validation only')
    args=ap.parse_args()
    old,_=load_run_configs(args.source_config); new,_=load_run_configs(args.target_config)
    if old.rate_split_mode!='simplex_scaled' or new.rate_split_mode!='per_user_fraction':
        raise ValueError('expected simplex_scaled -> per_user_fraction')
    a,b=old.to_dict(),new.to_dict(); a.pop('rate_split_mode');b.pop('rate_split_mode')
    if a!=b: raise ValueError('Only action representation may differ between environments')
    source=Path(args.source); out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    meta=json.loads((source/'meta.json').read_text())
    if meta.get('config_hash')!=config_hash(old): raise ValueError('source dataset hash mismatch')
    out.mkdir(parents=True)
    sizes={}
    for split in ['train','val','test']:
        with np.load(source/f'{split}.npz') as stored:
            data={k:stored[k].copy() for k in stored.files}
        count=len(data['observation'])
        if args.limit:
            data={k:(v[:args.limit] if v.ndim and len(v)==count else v) for k,v in data.items()}
        rf=data['oracle_rate_fractions']; rho=data['oracle_rho_r']
        u=np.zeros_like(rf)
        for i in range(len(rf)):
            active=data['oracle_mask'][i].astype(bool)
            total,private=split_payload_batch(np.array(old.payload_vector),active,rf[i:i+1],rho[i:i+1])
            u[i]=np.where(active,(np.array(old.payload_vector)-private[0])/np.array(old.payload_vector),0)
            check,dp=split_payload_batch(np.array(new.payload_vector),active,u[i:i+1],[1.],new.rate_split_mode)
            np.testing.assert_allclose(check,total,rtol=1e-6,atol=1e-5)
            np.testing.assert_allclose(dp,private,rtol=1e-6,atol=1e-5)
        data['oracle_rate_fractions']=u
        # Dataset schema compatibility only: ignored by per_user_fraction PHY.
        data['oracle_rho_r']=np.ones_like(rho)
        np.savez_compressed(out/f'{split}.npz',**data)
        sizes[split]=len(u)
    meta.update(source_dataset=str(source.resolve()),source_config_hash=config_hash(old),
        config_hash=config_hash(new),config=args.target_config,sizes=sizes,
        action_parameterization='direct_fraction_v1',
        label_conversion='Exact payload-preserving conversion, not new CEM solutions',
        rho_storage='Compatibility constant only; no actor rho action')
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
    print(out,sizes)
if __name__=='__main__':main()
