# Covid-19 Temperature/Mask Detection

Combine mask detector, temperature measurement, and social distancing evaluation on Jetson Xavier NX


## Prerequisites
- A jetson Xavier device with key based SSH access as local user
- An AWS account with CLI credentials configured

**Software Dependencies:**
- docker (on macOS, enable Experimental Features in Docker Desktop preferences)
- jq
- make
- python3
- python3-venv
- terraform

## Building Images

Each image can be built separately via a make target. The `build-all` target will build all docker dependencies. Once images are built they should be pushed to the remote docker repository so that they can be pulled by edge devices and the cloud image processor.

A dockerhub account is required to push images. This step is only required if you plan on building your own images. Otherwise, keep the docker repository environment variable set to the default value to pull previously built images.

Once the images are built, push them to dockerhub with the following command:

```
$ make push-all
```

Both of these steps can be performed at once as follows:

```
$ make build-all push-all
```

When using the build/push option, images will be labeled based on the CPU architecture so the [docker-compose](infrastructure/ansible/roles/edge_device/templates/docker-compose.yml.j2) file for the edge devices must be edited by adding the architecture to the image names.


## Multi-Architecture Builds

Building for multiple architectures simultaneously requires more configuration that the previous method. The benefit is that all docker images can be built on a single machine. In order to perform a multi-architecture build the following prerequisites must be met.

- Docker with buildx and experimental features enabled
- A buildx environment that can build both x86 and arm images
	`$ make buildx-setup`
- A dockerhub account since build and push happens in a single step

If all prerequisites are met, run the following command to build all images with x86 and ARM architectures and push to dockerhub.

```
$ make buildx-all
```

## Configuration

The `.env` file should be used to configure various aspects of the infrastructure deployment. Refer to the comments in the sample file for details on each configuration option.

## Deploy

Ansible and Terraform are used together to deploy the cloud infrastructure and configure the edge devices and cloud image processing server. Before running the make target to deploy you will likely need to initialize terraform. The Ansible dependencies will automatically be installed via a python virtual environment.

```
$ cd infrastructure/terraform && terraform init
```

Once terraform is initialized you can deploy the facial detection infrastructure

```
$ make deploy
```

## Teardown

To stop the image detector and cloud image processor run `make destroy`. This will leave all cloud infrastructure up with the exception of the EC2 instance used as the image processor. To destroy all cloud infrastructure (including the image capture S3 bucket) run `make destroy-all`.


## Results

The results of the facial detection pipeline can be viewed in the configured S3 bucket using the AWS CLI or at the following S3 static website.

```
http://<BUCKET_NAME>.s3-website-us-west-2.amazonaws.com/
```
