"""Conservative prelaunch cap using gross billed spend and all active project workers."""
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from pathlib import Path
import json,subprocess
from provision_worker import aws
HARD=Decimal('150');SOFT=Decimal('120')

def guard(config: dict,instance_price_cap: str,extra_cost_cap='1') -> dict:
    now=datetime.now(timezone.utc)
    period=f'Start={now:%Y-%m}-01,End={(now+timedelta(days=1)):%Y-%m-%d}'
    cost=aws('ce','get-cost-and-usage','--time-period',period,'--granularity','MONTHLY','--metrics','UnblendedCost','--filter',json.dumps({'Not':{'Dimensions':{'Key':'RECORD_TYPE','Values':['Credit','Refund']}}}))
    billed=sum((Decimal(r['Total']['UnblendedCost']['Amount'])for r in cost['ResultsByTime']),Decimal(0))
    gross=max(billed,Decimal(0))
    observed=aws('ec2','describe-instances','--filters','Name=tag:Project,Values=aml2026-phase5','Name=instance-state-name,Values=pending,running,stopping,stopped')
    active={i['InstanceId'] for group in observed['Reservations']for i in group['Instances']}
    ledgers=[json.loads(p.read_text())for p in Path('artifacts/cloud/phase5').glob('*/ledger.json')]
    known={x.get('instance_id')for x in ledgers}
    unknown=active-known
    if unknown:raise RuntimeError('Active project worker without local ledger; refusing launch')
    committed=Decimal(0);current=[]
    for entry in ledgers:
        if entry.get('instance_id') in active:
            ceiling=Decimal(str(entry['total_cost_ceiling_usd']));committed+=ceiling
            current.append({'run_id':entry['run_id'],'max_cost_usd':str(ceiling)})
    hours=Decimal(str(config['runtime_cap_minutes']))/Decimal(60)
    compute=Decimal(instance_price_cap)*hours
    proposed=compute+Decimal(extra_cost_cap)
    total=gross+committed+proposed
    if total>HARD or total>SOFT:raise RuntimeError(f'Project budget guard: gross {gross} + committed {committed} + proposed {proposed} exceeds soft/hard cap')
    if proposed>Decimal(str(config['total_cost_ceiling_usd'])):raise RuntimeError('Job planning ceiling understates computed worst case')
    record={'timestamp':now.isoformat(),'gross_month_to_date_usd':str(gross),'billing_estimated':cost['ResultsByTime'][-1].get('Estimated'),'active_workers':current,'running_max_committed_usd':str(committed),'new_hourly_price_cap_usd':instance_price_cap,'new_max_runtime_hours':str(hours),'new_max_compute_usd':str(compute),'new_storage_network_request_allowance_usd':extra_cost_cap,'new_max_total_usd':str(proposed),'project_worst_case_usd':str(total),'soft_cap_usd':str(SOFT),'hard_cap_usd':str(HARD),'credit_balance':'unverified'}
    return record
