Repository for holding environments and models for gridworld experiments with rllib, and deploying them with Ray & Tune on Google Cloud.

## Prerequisites

- pyvirtualenv
- google-api-python-client
- Service account keys for a GCP project: (https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
  - Save the credentials file inside `secrets/`.
- Create an `.env` file as follows, substituting values between <> with your own:
  ```sh
  GOOGLE_REGION=us-west1
  GOOGLE_AVAILABILITY_ZONE=us-west1-a
  GOOGLE_PROJECT_ID=< your project id >
  GOOGLE_APPLICATION_CREDENTIALS=./secrets/< name of downloaded credentials.json >
  ```

## Launch ray cluster on GCP

```sh
source .env  # load environment variables
envsubst < cluster.yaml.tmpl > cluster.yaml  # interpolate placeholders in cluster config template
ray up cluster.yaml
```

View the cluster's dashboard:

```sh
ray dashboard cluster.yaml
```

## Activate python env

TODO: Add instructions for usage without development-mode

Use python env from sibling ray directory (with dev symlinks)

```sh
export RAY_DEV_VENV='../ray/venv'
ln -s $RAY_DEV_VENV venv
source venv/bin/activate
```

## Run experiments

Run the notebook `run_experiment.ipynb`.
