resource "aws_s3_bucket" "diseasepredictor2025"{
    bucket = "diseasepredictor2025"

    tags = {
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
}
}

data "aws_caller_identity" "current" {}