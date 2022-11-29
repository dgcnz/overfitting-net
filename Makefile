
mlflow-server:
		mlflow server \
					--backend-store-uri sqlite:///mlflow.db \
					--serve-artifacts --artifacts-destination ./mlflow-artifacts \
									--host 0.0.0.0 \
											-p 5050
