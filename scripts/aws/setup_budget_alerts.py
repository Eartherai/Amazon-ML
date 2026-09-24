"""Create a separate $150 gross monthly project budget using an existing subscriber."""
import json
from pathlib import Path
from provision_worker import aws
NAME='AmazonMLChallenge2026-Gross-150'
ACCOUNT='634393786318'

def main():
 old={'NotificationType':'ACTUAL','ComparisonOperator':'GREATER_THAN','Threshold':85.0,'ThresholdType':'PERCENTAGE'}
 people=aws('budgets','describe-subscribers-for-notification','--account-id',ACCOUNT,'--budget-name','normalMy Monthly Cost Budget','--notification',json.dumps(old)).get('Subscribers',[])
 if not people:raise RuntimeError('No existing budget subscriber; refuse alert without recipient')
 existing={x['BudgetName']for x in aws('budgets','describe-budgets','--account-id',ACCOUNT)['Budgets']}
 if NAME in existing:raise RuntimeError('Project budget already exists; inspect before changing')
 budget={'BudgetName':NAME,'BudgetLimit':{'Amount':'150','Unit':'USD'},'TimeUnit':'MONTHLY','BudgetType':'COST','Metrics':['UnblendedCost'],'FilterExpression':{'Not':{'Dimensions':{'Key':'RECORD_TYPE','Values':['Credit','Refund']}}}}
 aws('budgets','create-budget','--account-id',ACCOUNT,'--budget',json.dumps(budget))
 thresholds=[25,50,75,100,125];receipt={'name':NAME,'gross':True,'limit_usd':150,'recipients':len(people),'notifications':[]}
 for dollar in thresholds:
  notification={'NotificationType':'ACTUAL','ComparisonOperator':'GREATER_THAN','Threshold':100*dollar/150,'ThresholdType':'PERCENTAGE'}
  aws('budgets','create-notification','--account-id',ACCOUNT,'--budget-name',NAME,'--notification',json.dumps(notification),'--subscribers',json.dumps(people))
  receipt['notifications'].append({'dollars':dollar,'percentage':notification['Threshold']})
 p=Path('artifacts/cloud/phase5/budget-alerts.json');p.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
