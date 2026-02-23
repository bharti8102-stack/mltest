## First Project on Machine Learning
import dagshub
dagshub.init(repo_owner='bharti8102', repo_name='mltest', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
