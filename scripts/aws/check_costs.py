"""Read current-month AWS costs/budgets; billed credits are not credit balance."""
from datetime import datetime,timezone,timedelta
import json,subprocess
from pathlib import Path
from check_capacity import call
now=datetime.now(timezone.utc);print('UTC:',now.isoformat())
profile='amamzon_01_a1_0'
account=subprocess.check_output(['aws','--profile',profile,'sts','get-caller-identity','--query','Account','--output','text'],text=True).strip()
results={'timestamp':now.isoformat(),'balance':'Not verified; Cost Explorer credits are applied charges, not available balance.'}
results['budgets']=call(profile,['budgets','describe-budgets','--account-id',account])
results['costs']=call(profile,['ce','get-cost-and-usage','--time-period',f'Start={now:%Y-%m}-01,End={(now+timedelta(days=1)):%Y-%m-%d}','--granularity','MONTHLY','--metrics','UnblendedCost','--group-by','Type=DIMENSION,Key=SERVICE'])
p=Path('artifacts/cloud/phase5')/now.strftime('costs-%Y%m%dT%H%M%SZ.json');p.write_text(json.dumps(results,indent=2));print(json.dumps(results))
