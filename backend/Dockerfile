# syntax=docker/dockerfile:1

FROM golang:1.22-alpine AS build
WORKDIR /app

# go.sum is not committed yet (no Go toolchain on the dev host);
# `go mod download` fetches deps and writes go.sum inside the image.
COPY go.mod ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o /bin/server ./cmd/server

FROM alpine:3.20
RUN adduser -D -u 10001 app
USER app
COPY --from=build /bin/server /bin/server
EXPOSE 8080
ENTRYPOINT ["/bin/server"]
