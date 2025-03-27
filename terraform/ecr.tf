resource "aws_ecr_repository" "disease_predictor" {
  name                 = "disease_predictor"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}