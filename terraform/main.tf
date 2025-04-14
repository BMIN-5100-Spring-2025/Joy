data "aws_s3_bucket" "diseasepredictor2025"{
    bucket = "diseasepredictor2025"

    //tags = {
    //Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
//}
}

data "aws_caller_identity" "current" {}

# Define role
data "aws_iam_policy_document" "assume_role_policy_dp" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}
# Create role
resource "aws_iam_role" "execution_role_dp" {
  name               = "execution_role_dp"
  //path               = "/system/"
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy_dp.json
}
# Attach pre-defined policy 
resource "aws_iam_role_policy_attachment" "execution_pre_attach" {
  role       = aws_iam_role.execution_role_dp.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
# Define policy 
data "aws_iam_policy_document" "execution_policy_dp" {
  statement {
    effect    = "Allow"
    actions   = ["ec2:Describe*",
        "ec2:DescribeNetworkInterfaces",
        "ec2:CreateNetworkInterface",
        "ec2:AttachNetworkInterface",
        "ec2:DeleteNetworkInterface",
        "ec2:AssignPrivateIpAddresses",
        "ec2:UnassignPrivateIpAddresses"]
    resources = ["*"]
  }
}
# Create policy
resource "aws_iam_policy" "execution_policy_dp" {
  name        = "execution_policy_dp"
  description = "An execution policy"
  policy      = data.aws_iam_policy_document.execution_policy_dp.json
}
# Attach policy
resource "aws_iam_role_policy_attachment" "execution_attach" {
  role       = aws_iam_role.execution_role_dp.name
  policy_arn = aws_iam_policy.execution_policy_dp.arn
}


# task_role definition 
data "aws_iam_policy_document" "task_role_policy_dp" {
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
resource "aws_iam_role" "task_role_dp" {
    name = "task_role_dp"
    assume_role_policy = data.aws_iam_policy_document.task_role_policy_dp.json
}
# task_role policy define
data "aws_iam_policy_document" "task_policy_dp" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:*"
    ]
    resources = [
      data.aws_s3_bucket.diseasepredictor2025.arn,
      "${data.aws_s3_bucket.diseasepredictor2025.arn}/*"
    ]
  }
}
# task_role policy create 
resource "aws_iam_policy" "task_policy_dp" {
  name        = "task_policy_dp"
  description = "A task policy"
  policy      = data.aws_iam_policy_document.task_policy_dp.json
}
# task policy attachment
resource "aws_iam_role_policy_attachment" "task_attach" {
  role       = aws_iam_role.task_role_dp.name
  policy_arn = aws_iam_policy.task_policy_dp.arn
}



# task definition
resource "aws_ecs_task_definition" "disease_predictor" {
  family                   = "disease_predictor"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  memory                   = 2048
  cpu = "512"
  execution_role_arn = aws_iam_role.execution_role_dp.arn
  task_role_arn = aws_iam_role.task_role_dp.arn
  container_definitions    = jsonencode([
    {
      name  = "disease_predictor"
      image = "${aws_ecr_repository.disease_predictor.repository_url}:v1.6"
      environment = [
        { name = "s3_arn", value = data.aws_s3_bucket.diseasepredictor2025.arn },
        { name = "MODE", value = "s3"}
      ],
      logConfiguration = {
          "logDriver" = "awslogs",
          "options" = {
            "awslogs-group" = "disease_predictor",
            //"awslogs-create-group" = "true",
            "awslogs-region" = "us-east-1",
            "awslogs-stream-prefix" = "ecs"
          }
      }
    }
  ])
}