# Install AWS Command Line Interface
# https://aws.amazon.com/cli/
apk add --update python python-dev py-pip
pip install awscli --upgrade



# Set AWS config variables used during the AWS get-login command below
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


# The AWS registry now has our new container, but our cluster isn't aware that a new version
# of the container is available. We need to create an updated task definition. Task definitions
# always have a version number. When we register a task definition using a name that already
# exists, AWS automatically increments the previously used version number for the task
# definition with that same name and uses it here. Note that we also define CPU and memory
# requirements here and give it a JSON file describing our task definition that I've saved
# to my repository in a aws/ directory.
#aws ecs register-task-definition --family ask-detection-$CI_ENVIRONMENT_SLUG	 --requires-compatibilities FARGATE --cpu 2048 --memory 4096 --cli-input-json file://aws/ask-detection-task-definition-$CI_ENVIRONMENT_SLUG.json --region $AWS_REGION

# Tell our service to use the latest version of task definition.
#aws ecs update-service --cluster ask-detection-$CI_ENVIRONMENT_SLUG --service ask_detection_service --task-definition ask-detection-$CI_ENVIRONMENT_SLUG --region $AWS_REGION

# Attempt to tell service to simply pull the new version of the container image
aws ecs update-service --cluster ask-detection-cluster --service ask-detection-with-balancer --force-new-deployment --region $AWS_REGION
