resource "aws_vpc" "w251" {
  cidr_block          = "10.0.0.0/16"
  instance_tenancy    = "default"

  tags = {
    Name = "w251"
  }
}

resource "aws_internet_gateway" "egress" {
  vpc_id = aws_vpc.w251.id
}

resource "aws_route" "r" {
  route_table_id         = aws_vpc.w251.main_route_table_id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.egress.id
}

resource "aws_subnet" "w251" {
  vpc_id                  = aws_vpc.w251.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true

  tags = {
    Name = "w251"
  }
}

