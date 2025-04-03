data "aws_s3_bucket" "diseasepredictor2025"{
    bucket = "diseasepredictor2025"

    //tags = {
    //Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
//}
}

data "aws_caller_identity" "current" {}

# Define role
data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}
# Create role
resource "aws_iam_role" "execution_role" {
  name               = "execution_role"
  //path               = "/system/"
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
}
# Attach pre-defined policy 
resource "aws_iam_role_policy_attachment" "execution_pre_attach" {
  role       = aws_iam_role.execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
# Define policy 
data "aws_iam_policy_document" "execution_policy" {
  statement {
    effect    = "Allow"
    actions   = ["ec2:Describe*"]
    resources = ["*"]
  }
}
# Create policy
resource "aws_iam_policy" "execution_policy" {
  name        = "execution_policy"
  description = "An execution policy"
  policy      = data.aws_iam_policy_document.execution_policy.json
}
# Attach policy
resource "aws_iam_role_policy_attachment" "execution_attach" {
  role       = aws_iam_role.execution_role.name
  policy_arn = aws_iam_policy.execution_policy.arn
}


# task_role definition 
data "aws_iam_policy_document" "task_role_policy" {
  statement {
    effect = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# task_role Create
resource "aws_iam_role" "task_role" {
    name = "task_role"
    assume_role_policy = data.aws_iam_policy_document.task_role_policy.json
}
# task_role policy define
data "aws_iam_policy_document" "task_policy" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      data.aws_s3_bucket.diseasepredictor2025.arn,
      "${data.aws_s3_bucket.diseasepredictor2025.arn}/*"
    ]
  }
}
# task_role policy create 
resource "aws_iam_policy" "task_policy" {
  name        = "task_policy"
  description = "A task policy"
  policy      = data.aws_iam_policy_document.task_policy.json
}
# task policy attachment
resource "aws_iam_role_policy_attachment" "task_attach" {
  role       = aws_iam_role.task_role.name
  policy_arn = aws_iam_policy.task_policy.arn
}



# task definition
resource "aws_ecs_task_definition" "disease_predictor" {
  family                   = "disease_predictor"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  memory                   = 2048
  execution_role_arn = aws_iam_role.execution_role.arn
  task_role_arn = aws_iam_role.task_role.arn
  container_definitions    = jsonencode([
    {
      name  = "disease_predictor"
      image = "061051226319.dkr.ecr.us-east-1.amazonaws.com/disease_predictor:v1"
      environment = [
        { name = "s3_arn", value = data.aws_s3_bucket.diseasepredictor2025.arn }
      ]
    }
  ])
}