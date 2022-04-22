
# exit when any command fails
set -e

# Ensure the models on DVC are being deployed
dvc pull data/models.dvc

# TODO: get values from env file
export MODELS=$(
        python -c \
        "from pathlib import Path; \
        print(' '.join([p.stem for p in Path('data/models').glob('*.pt')]))"
)

docker build . --build-arg MODELS="$MODELS" -t $TAG
docker push $TAG
gcloud run deploy ${PROJECT} --image ${TAG}:latest \
        --memory=8Gi \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated \
        --concurrency 10 \
        --port 8080

gcloud run deploy ${PROJECT}-management-api --image ${TAG}:latest \
        --memory=4Gi \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated \
        --port 8081

gcloud functions deploy trigger-inference \
    --source=src/trigger_inference_function \
    --trigger-bucket=$BUCKET \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=trigger \
    --set-env-vars MODELS="$MODELS",INFERENCE_HOST="$URL" \
    --timeout=300s
