variable "instance_type" {}
variable "key_name" {}
variable "region" {}

provider "aws" {
  region = var.region
}

output "region" {
  value = var.region
}

data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-bionic-18.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

data "external" "myipaddr" {
  program = ["bash", "-c", "curl -s 'https://api.ipify.org?format=json'"]
}

output "my_public_ip" {
  value = data.external.myipaddr.result.ip
}

resource "aws_security_group" "w251_image_server" {
  name        = "sg_w251_image_server"
  description = "Allow SSH and MQTT inbound traffic"
  vpc_id      = aws_vpc.w251.id

  ingress {
    description = "SSH from public IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${data.external.myipaddr.result.ip}/32"]
  }

  ingress {
    description = "MQTT from public IP"
    from_port   = 1883
    to_port     = 1883
    protocol    = "tcp"
    cidr_blocks = ["${data.external.myipaddr.result.ip}/32"]
  }

  ingress {
    description = "Metabase from public IP"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["${data.external.myipaddr.result.ip}/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sg_w251_image_server"
  }
}

resource "aws_instance" "w251_image_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.w251_image_server.name
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.w251_image_server.id]
  subnet_id              = aws_subnet.w251.id

  tags = {
    Name = "w251-image-server"
  }
}

output "image_server_public_ip" {
  value = aws_instance.w251_image_server.public_ip
}
