variable "bucket_name" {}

output "s3_bucket_name" {
  value = var.bucket_name
}

resource "aws_s3_bucket" "image_bucket" {
  bucket        = var.bucket_name
  acl           = "public-read"
  force_destroy = true

  website {
    index_document = "index.html"
    error_document = "error.html"
  }

  policy = <<-EOF
{
  "Id": "Policy1599427644637",
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Stmt1599427639571",
      "Action": [
        "s3:GetObject"
      ],
      "Effect": "Allow",
      "Resource": "arn:aws:s3:::${var.bucket_name}/*",
      "Principal": "*"
    }
  ]
}
EOF

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
  }
}

