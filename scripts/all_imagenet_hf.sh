search_dir=data/video/6-75
MLFLOW_HOST="${MLFLOW_HOST}"
MLFLOW_EXPERIMENT_ID="${MLFLOW_EXPERIMENT_ID}"
echo $MLFLOW_HOST
echo $MLFLOW_EXPERIMENT_ID
for entry in "$search_dir"/*
do
  echo "$entry"
  MLFLOW_EXPERIMENT_ID=$MLFLOW_EXPERIMENT_ID MLFLOW_HOST=$MLFLOW_HOST python -m scripts.experiment_hf $entry --confidence 0.1 --weight_decay 0.2 --max_lr 0.4
done
