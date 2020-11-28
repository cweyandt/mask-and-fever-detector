# 1. Introduction
This folder contains files for creating a docker image for training mask classifier on AWS cloud. Once started, it reads the face data from the configured S3 bucket and trains the model. Output model will be written to the output S3 bucket and model dynamodb will be updated. 

Client can query and download the model by using the following API gateway interface:

# 2. Testing environment
AWS region: us-east-2
EC2 instance AMI: ami-01bd6a1621a6968d7
EC2 instance type: p3.2xlarge
Input S3 bucket: covid19-detector
output S3 bucket: covid19-models

# 3. How to train the model 
1. Start up EC2 instance
2. Attach IAM role so that it can read/write to S3 buckets.
3. Clone this repo on the EC2 instance
4. Update environment variables to point to the correct bucket and dynamodb names.
5. Star the container
