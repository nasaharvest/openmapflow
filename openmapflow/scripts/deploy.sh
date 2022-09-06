#####################################################################################################
# This script deploys all the resources necessary to make maps on Google Cloud with OpenMapFlow
#####################################################################################################

set -e # Exit when any command fails

echo "1/8 Setting OpenMapFlow environment variables"
export $(
        python -c \
        "from openmapflow.config import deploy_env_variables; \
        print(deploy_env_variables())"
        )

env | grep OPENMAPFLOW


echo "2/8 Ensuring latest models are available for deployment"
dvc pull "$OPENMAPFLOW_MODELS_DIR".dvc -f

if [[ -z "$OPENMAPFLOW_MODELS" ]]; then
        export OPENMAPFLOW_MODELS=$(
                python -c \
                "from openmapflow.config import get_model_names_as_str; \
                print(get_model_names_as_str())"
        )
fi

echo "MODELS: $OPENMAPFLOW_MODELS"

echo "3/8 Enable Google Cloud APIs: Artifact Registry, Cloud Run, Cloud Functions"
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable earthengine.googleapis.com
gcloud services enable run.googleapis.com


echo "4/8 Create Google Cloud Buckets if they don't exist"
for BUCKET in $OPENMAPFLOW_GCLOUD_BUCKET_LABELED_EO \
        $OPENMAPFLOW_GCLOUD_BUCKET_INFERENCE_EO \
        $OPENMAPFLOW_GCLOUD_BUCKET_PREDS \
        $OPENMAPFLOW_GCLOUD_BUCKET_PREDS_MERGED
do
        if [ "$(gsutil ls -b gs://$BUCKET)" ]; then
                echo "gs://$BUCKET already exists"
        else
                gsutil mb -l $OPENMAPFLOW_GCLOUD_LOCATION gs://$BUCKET
        fi
done


echo "5/8 Checking if Artifact Registry needs to be created for storing OpenMapFlow docker images"
if [ -z "$(gcloud artifacts repositories list --format='get(name)' --filter "$OPENMAPFLOW_PROJECT")" ]; then
        gcloud artifacts repositories create "$OPENMAPFLOW_PROJECT" \
        --location "$OPENMAPFLOW_GCLOUD_LOCATION" \
        --repository-format docker
fi

echo "6/8 Build and push inference docker image to Google Cloud artifact registry"
gcloud auth configure-docker "${OPENMAPFLOW_GCLOUD_LOCATION}"-docker.pkg.dev
docker build . \
        -f "$OPENMAPFLOW_LIBRARY_DIR"/Dockerfile \
        --build-arg MODELS="$OPENMAPFLOW_MODELS" \
        --build-arg MODELS_DIR="$OPENMAPFLOW_MODELS_DIR" \
        --build-arg DEST_BUCKET="$OPENMAPFLOW_GCLOUD_BUCKET_PREDS" \
        --build-arg VERSION="$OPENMAPFLOW_VERSION" \
        -t "$OPENMAPFLOW_DOCKER_TAG"

docker push "$OPENMAPFLOW_DOCKER_TAG"


echo "7/8 Deploy inference docker image to Google Cloud Run"
echo "Deploying prediction server on port 8080"
gcloud run deploy "$OPENMAPFLOW_PROJECT" --image "$OPENMAPFLOW_DOCKER_TAG":latest \
        --cpu=4 \
        --memory=8Gi \
        --platform=managed \
        --region="$OPENMAPFLOW_GCLOUD_LOCATION" \
        --allow-unauthenticated \
        --concurrency 10 \
        --max-instances 1000 \
        --port 8080

echo "Deploying model list server on port 8081"
gcloud run deploy "$OPENMAPFLOW_PROJECT"-management-api --image "$OPENMAPFLOW_DOCKER_TAG":latest \
        --memory=4Gi \
        --platform=managed \
        --region="$OPENMAPFLOW_GCLOUD_LOCATION" \
        --allow-unauthenticated \
        --port 8081



echo "8/8 Deploy inference trigger as a Google Cloud Function"
export OPENMAPFLOW_URL=$(gcloud run services list --platform managed --filter $OPENMAPFLOW_PROJECT --limit 1 --format='get(URL)')

gcloud functions deploy trigger-"$OPENMAPFLOW_PROJECT" \
    --source="$OPENMAPFLOW_LIBRARY_DIR"/trigger_inference_function \
    --trigger-bucket="$OPENMAPFLOW_GCLOUD_BUCKET_INFERENCE_EO" \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=trigger \
    --set-env-vars MODELS="$OPENMAPFLOW_MODELS",INFERENCE_HOST="$OPENMAPFLOW_URL" 
