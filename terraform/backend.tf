terraform {
  backend "s3" {
    bucket = "diseasepredictor2025"
    key    = "enhuz@seas.upenn.edu-diseasepredictor/terraform.tfstate"
    region = "us-east-1"
  }
}