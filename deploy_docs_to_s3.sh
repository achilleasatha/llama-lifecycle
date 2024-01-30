#!/bin/bash

# Copy the latest README.md file to the /docs directory
cp README.md docs/README.md

# Build the documentation using MkDocs
mkdocs build

# Read configuration from config.json
S3_BUCKET_NAME=$(jq -r '.s3_bucket_name' config.json)
AWS_REGION=$(jq -r '.aws_region' config.json)

# Check if the S3 bucket exists
if aws s3api head-bucket --bucket "$S3_BUCKET_NAME" 2>/dev/null; then
    echo "S3 bucket $S3_BUCKET_NAME already exists."
else
    # Create the S3 bucket if it doesn't exist
    aws s3api create-bucket --bucket "$S3_BUCKET_NAME" --region "$AWS_REGION" --create-bucket-configuration LocationConstraint="$AWS_REGION"
    echo "S3 bucket $S3_BUCKET_NAME created."
fi

# Set the bucket policy to make contents publicly readable
aws s3api put-bucket-policy --bucket "$S3_BUCKET_NAME" --policy '{
  "Version":"2012-10-17",
  "Statement":[{
    "Sid":"PublicReadGetObject",
    "Effect":"Allow",
    "Principal": "*",
    "Action":["s3:GetObject"],
    "Resource":["arn:aws:s3:::'"$S3_BUCKET_NAME"'/*"]
  }]
}'

#aws s3api delete-public-access-block --bucket $S3_BUCKET_NAME
#aws s3api put-bucket-policy --bucket $S3_BUCKET_NAME --policy "$BUCKET_POLICY" --region $AWS_REGION

# Push the generated static files to S3 bucket with public-read ACL
aws s3 sync site/ "s3://$S3_BUCKET_NAME" --delete #--acl public-read

# Enable static website hosting
aws s3 website s3://$S3_BUCKET_NAME --index-document index.html

echo "Documentation deployed to S3 bucket $S3_BUCKET_NAME in $AWS_REGION. Remember to set public-read access if required."
