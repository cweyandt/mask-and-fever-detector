package main

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	mqtt "github.com/eclipse/paho.mqtt.golang"
)

var (
	client     mqtt.Client
	awsSession *session.Session
	topic      = "#" // subscribe to all topics
	S3_REGION  = os.Getenv("S3_REGION")
	S3_BUCKET  = os.Getenv("S3_BUCKET")
)

func init() {
	localMQTT, err := url.Parse(os.Getenv("LOCALMQTT_URL"))
	if err != nil {
		log.Fatal(err)
	}
	client = connect("sub", localMQTT)

	// Create a single AWS session (we can re use this if we're uploading many files)
	awsSession, err = session.NewSession(&aws.Config{Region: aws.String(S3_REGION)})
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	client.Subscribe(topic, 0, processMessage)
	select {}
}

func connect(clientId string, uri *url.URL) mqtt.Client {
	opts := createClientOptions(clientId, uri)
	client := mqtt.NewClient(opts)
	token := client.Connect()
	for !token.WaitTimeout(3 * time.Second) {
	}
	if err := token.Error(); err != nil {
		log.Fatal(err)
	}
	return client
}

func createClientOptions(clientId string, uri *url.URL) *mqtt.ClientOptions {
	opts := mqtt.NewClientOptions()
	opts.AddBroker(fmt.Sprintf("tcp://%s", uri.Host))
	opts.SetUsername(uri.User.Username())
	password, _ := uri.User.Password()
	opts.SetPassword(password)
	opts.SetClientID(clientId)
	return opts
}

func processMessage(client mqtt.Client, msg mqtt.Message) {
	// fmt.Printf("* [%s] %s\n", msg.Topic(), string(msg.Payload()))
	fmt.Printf("recieved message from topic: %s\n", msg.Topic())
	if err := saveToS3(msg.Topic(), msg.Payload()); err != nil {
		fmt.Printf("error saving to s3: %s\n", err.Error())
	}
}

// saveToS3 will upload a byte array to S3
func saveToS3(topic string, blob []byte) error {
	topicSplit := strings.Split(topic, "/")
	ext := ""
	if len(topicSplit) > 1 {
		topic = strings.Join(topicSplit[:len(topicSplit)-1], "/")
		ext = "." + topicSplit[len(topicSplit)-1]
	}
	key := fmt.Sprintf("%s/%d%s", topic, time.Now().UnixNano(), ext)
	fmt.Printf("saving to s3: s3://%s/%s\n", S3_BUCKET, key)
	_, err := s3.New(awsSession).PutObject(&s3.PutObjectInput{
		Bucket:             aws.String(S3_BUCKET),
		Key:                aws.String(key),
		Body:               bytes.NewReader(blob),
		ContentLength:      aws.Int64(int64(len(blob))),
		ContentType:        aws.String(http.DetectContentType(blob)),
		ContentDisposition: aws.String("attachment"),
	})
	return err
}
