FROM golang:1.15 AS builder

WORKDIR /go/src/mids.w251.edu/image-processor

COPY . .
RUN CGO_ENABLED=0 go build -o /tmp/image-processor main.go

FROM alpine

COPY --from=builder /tmp/image-processor /bin/image-processor

# Add docker-compose-wait tool -------------------
ENV WAIT_VERSION 2.7.3
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/$WAIT_VERSION/wait /wait
RUN chmod +x /wait

COPY database.yml .
ENTRYPOINT ["/bin/image-processor"]
