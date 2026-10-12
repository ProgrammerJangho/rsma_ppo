"""Reproducible bounded pilot: unchanged PHY, paired CEM, PPO and near-zero audit."""
from pathlib import Path
import sys, json, argparse, time
from dataclasses import replace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rsma_aoi.utils.config import load_run_configs
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.agents.ppo import PPOTrainer
from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy


def phy_arrays(action, direct):
    """Convert the public action schema to the PHY's compact internal arrays."""
    if not direct:
        return (action['power_fractions'], action['rate_fractions'],
                float(action['common_payload_scale'][0]))
    alpha=float(action['common_power_fraction'][0])
    q=np.asarray(action['private_power_fractions'],float)
    pf=np.r_[alpha,(1-alpha)*q]
    return pf,np.asarray(action['common_payload_fractions'],float),1.0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=8192)
    parser.add_argument('--out',default='results/direct_fraction_pilot_v1')
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=ROOT/args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    old,pc=load_run_configs(ROOT/'configs/thesis_rician_mixed_geometry_snr_csit02.yaml')
    new,npc=load_run_configs(ROOT/'configs/thesis_direct_fraction.yaml')
    # Continuous-bit normal approximation is deliberately left unchanged.
    env=AoIRSMAEnv(new); env.reset(seed=123)
    phy=RSMAPhysicalLayer(new); active=np.ones(new.num_users,bool)
    h=np.array([[1.,0.5j],[.9,.3+.2j]])
    beams=phy.rsma_beams(h,active)
    boundary=phy.evaluate_allocations(h,active,[[0,.5,.5]],[[0,0]],[1],beams=beams)
    rows=[]
    for exponent in [0.5,1.,2.]:
        for eps in [1e-2,1e-4,1e-6,1e-8]:
            result=phy.evaluate_allocations(h,active,[[eps,(1-eps)/2,(1-eps)/2]],
                [[eps**exponent]*2],[1],beams=beams)
            rows.append(dict(alpha=eps,u=eps**exponent,payload_exponent=exponent,
                common_bits=float(result['common_payload'][0]),
                common_bler=result['common_bler'][0].tolist(),
                success=result['success_prob'][0].tolist(),
                sdma_success=boundary['success_prob'][0].tolist()))
    (out/'near_zero.json').write_text(json.dumps(rows,indent=2))
    print('NEAR_ZERO',rows[-1],flush=True)

    # Hold state, AoI, beam family and CEM candidate budget fixed across representations.
    cem=[]
    for angle in [5,20,60]:
        for snr in [10,15]:
            oc=replace(old,user_angle_sets_deg=None,user_angles_deg=[-angle/2,angle/2],snr_db_values=None,snr_db=snr)
            nc=replace(oc,rate_split_mode='per_user_fraction')
            probe=AoIRSMAEnv(oc)
            a=CEMAllocationPolicy(oc,'rsma_safeguarded')
            b=CEMAllocationPolicy(nc,'rsma_safeguarded')
            for state_seed in range(16):
                obs,_=probe.reset(seed=1000+state_seed)
                state=probe.get_state()
                state['aoi']=np.array([1+state_seed%7,1+(state_seed*3)%7],float)
                values=[]
                for cfg,policy in [(oc,a),(nc,b)]:
                    policy.reset(state_seed)
                    action=policy.act(obs[None],[state])[0]
                    pf,phi,rho=phy_arrays(action,cfg.rate_split_mode=='per_user_fraction')
                    link=RSMAPhysicalLayer(cfg)
                    link.total_power=state['total_power']
                    res=link.evaluate_allocations(probe._h_true,active,pf[None],
                        phi[None],np.array([rho]),h_hat=state['h_hat'])
                    next_aoi=(state['aoi']+1)-state['aoi']*res['success_prob'][0]
                    values.append(float(next_aoi.mean()))
                cem.append(dict(angle=angle,snr=snr,state_seed=state_seed,old_expected_aoi=values[0],new_expected_aoi=values[1]))
            print('CEM',angle,snr,'done',flush=True)
    (out/'cem_paired.json').write_text(json.dumps(cem,indent=2))
    evaluations=[]
    for seed in range(3):
        for name,cfg,agent in [('legacy',old,pc),('direct',new,npc)]:
            run=ROOT/f'runs/{out.name}/{name}_seed{seed}'
            if (run/'checkpoint.pt').exists():
                raise FileExistsError(run)
            agent=replace(agent,total_timesteps=args.steps,checkpoint_interval=0,eval_interval=0)
            trainer=PPOTrainer(cfg,agent,run,seed=seed)
            start=time.monotonic()
            trainer.train()
            policy=trainer.make_policy(deterministic=True)
            print('TRAINED',name,seed,round(time.monotonic()-start,1),flush=True)
            for angle in [5,20,60]:
                for snr in [10,15]:
                    ec=replace(cfg,user_angle_sets_deg=None,user_angles_deg=[-angle/2,angle/2],snr_db_values=None,snr_db=snr)
                    for projected in [False,True]:
                        for episode in range(2):
                            test=AoIRSMAEnv(ec); obs,_=test.reset(seed=8000+episode)
                            ages=[]; commons=[]; powers=[]; probs=[]; blers=[]; priv_blers=[]
                            burst=np.zeros(2,int); maxburst=np.zeros(2,int)
                            for step in range(512):
                                action=policy.act(obs[None],[test.get_state()])[0]
                                if projected:
                                    if name=='direct':
                                        action['common_power_fraction']=np.zeros(1)
                                        action['common_payload_fractions']=np.zeros(2)
                                    else:
                                        pf=action['power_fractions']; private=pf[1:]/pf[1:].sum()
                                        action['power_fractions']=np.r_[0.,private]
                                        action['common_payload_scale']=np.zeros(1)
                                obs,reward,term,trunc,info=test.step(action)
                                burst=np.where(info['ack']>.5,0,burst+1); maxburst=np.maximum(maxburst,burst)
                                ages.append(info['aoi']); commons.append(info['common_payload'])
                                powers.append(info['common_power']/test.total_power)
                                probs.append(info['success_probabilities']); blers.append(info['common_bler']); priv_blers.append(info['private_bler'])
                            ages=np.array(ages)
                            evaluations.append(dict(policy=name,seed=seed,angle=angle,snr=snr,projected=projected,
                                episode=episode,mean_aoi=float(ages.mean()),user_aoi=ages.mean(0).tolist(),
                                p95_aoi=float(np.quantile(ages,.95)),max_failure_burst=maxburst.tolist(),
                                common_bits=float(np.mean(commons)),alpha=float(np.mean(powers)),
                                success=np.mean(probs,0).tolist(),common_bler=np.mean(blers,0).tolist(),private_bler=np.mean(priv_blers,0).tolist()))
                            test.close()
            trainer.close()
            (out/'ppo_paired.json').write_text(json.dumps(evaluations,indent=2))
    print('COMPLETE',out,flush=True)
if __name__=='__main__': main()
