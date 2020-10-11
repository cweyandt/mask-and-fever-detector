FROM golang:1.15 AS builder

WORKDIR /go/src/mids.w251.edu/image-processor

COPY . .
RUN CGO_ENABLED=0 go build -o /tmp/image-processor main.go

FROM alpine

COPY --from=builder /tmp/image-processor /bin/image-processor
ENTRYPOINT ["/bin/image-processor"]
