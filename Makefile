
mlflow-server:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow-artifact \
		--host 0.0.0.0 \
		-p 5050
