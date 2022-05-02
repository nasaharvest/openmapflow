
# exit when any command fails
set -e

export $(
        python -c \
        "from openmapflow.config import deploy_env_variables; \
        print(deploy_env_variables())"
        )
echo "PROJECT: $PROJECT"
echo "MODELS_DIR: $MODELS_DIR"
echo "GCLOUD_PROJECT: $GCLOUD_PROJECT"
echo "GCLOUD_LOCATION: $GCLOUD_LOCATION"
echo "TAG: $TAG"
echo "OPENMAPFLOW_DIR: $OPENMAPFLOW_DIR"

# Ensure the models on DVC are being deployed
dvc pull $MODELS_DIR.dvc

export MODELS=$(
        python -c \
        "from openmapflow.config import get_model_names_as_str; \
        print(get_model_names_as_str())"
)
echo "MODELS: $MODELS"

# Enable artifact registry
gcloud services enable artifactregistry.googleapis.com

if [ -z "$(gcloud artifacts repositories list --format='get(name)' --filter $PROJECT)" ]; then
        gcloud artifacts repositories create $PROJECT --location $GCLOUD_LOCATION --repository-format docker
fi

docker build . \
        -f $OPENMAPFLOW_DIR/Dockerfile \
        --build-arg MODELS="$MODELS" \
        --build-arg MODELS_DIR="$MODELS_DIR" \
        -t $TAG

docker push $TAG

gcloud run deploy $PROJECT --image $TAG:latest \
        --cpu=4 \
        --memory=8Gi \
        --platform=managed \
        --region=$GCLOUD_LOCATION \
        --allow-unauthenticated \
        --concurrency 10 \
        --port 8080

gcloud run deploy $PROJECT-management-api --image $TAG:latest \
        --memory=4Gi \
        --platform=managed \
        --region=$GCLOUD_LOCATION \
        --allow-unauthenticated \
        --port 8081

# gcloud functions deploy trigger-inference \
#     --source=src/trigger_inference_function \
#     --trigger-bucket=$BUCKET \
#     --allow-unauthenticated \
#     --runtime=python39 \
#     --entry-point=trigger \
#     --set-env-vars MODELS="$MODELS",INFERENCE_HOST="$URL" \
#     --timeout=300s
