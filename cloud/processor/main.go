package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/gobuffalo/pop"
	"github.com/gofrs/uuid"
)

var (
	client     mqtt.Client
	awsSession *session.Session
	topic      = "#" // subscribe to all topics
	S3_REGION  = os.Getenv("S3_REGION")
	S3_BUCKET  = os.Getenv("S3_BUCKET")
	db         *pop.Connection
)

type MQTTMessage struct {
	ID              uuid.UUID `db:"id"`
	Topic           string    `db:"topic"`
	DetectionType   string    `json:"detection_type" db:"detection_type"`
	ImageEncoding   string    `json:"image_encoding" db:"-"`
	HasFever        bool      `json:"has_fever" db:"has_fever"`
	Base64Frame     string    `json:"frame" db:"-"`
	Base64FullFrame string    `json:"full_frame" db:"-"`
	CreatedAt       time.Time `db:"created_at"`
	UpdatedAt       time.Time `db:"updated_at"`
}

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

	db_url := os.Getenv("DATABASE_URL")
	fmt.Printf("Databse URL: %s", db_url)
	if db_url != "" {
		db, err = pop.Connect("")
		if err != nil {
			log.Fatal(err)
		}
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
	fmt.Printf("received message from topic: %s\n", msg.Topic())
	m := MQTTMessage{}
	if err := json.Unmarshal(msg.Payload(), &m); err != nil {
		fmt.Printf("error processing message: %s\n", err.Error())
	}

	m.Topic = msg.Topic()
	if err := m.Save(); err != nil {
		fmt.Printf("error saving message: %s\n", err.Error())
	}
}

func (m MQTTMessage) TableName() string {
	return "message_stats"
}

func (m *MQTTMessage) Save() error {
	if db != nil {
		fmt.Printf("saving to postgres: %s\n", m.DetectionType)
		if err := db.Create(m); err != nil {
			return err
		}
	}
	return m.saveToS3()
}

func (m *MQTTMessage) saveToS3() error {
  now := time.Now().UnixNano()
	key := fmt.Sprintf("%s/%d.%s", m.DetectionType, now, m.ImageEncoding)
	if err := saveToS3(key, m.Base64Frame); err != nil {
		return err
	}

	key = fmt.Sprintf("%s/%d_full.%s", m.DetectionType, now, m.ImageEncoding)
	return saveToS3(key, m.Base64FullFrame)
}

func saveToS3(key, image string) error {
	if image == "" {
		return nil
	}
	img, err := base64.StdEncoding.DecodeString(image)
	if err != nil {
		return fmt.Errorf("unable to decoding image: %s ", err.Error())
	}

	fmt.Printf("saving to s3: s3://%s/%s\n", S3_BUCKET, key)
	_, err = s3.New(awsSession).PutObject(&s3.PutObjectInput{
		Bucket:             aws.String(S3_BUCKET),
		Key:                aws.String(key),
		Body:               bytes.NewReader(img),
		ContentLength:      aws.Int64(int64(len(img))),
		ContentType:        aws.String(http.DetectContentType(img)),
		ContentDisposition: aws.String("attachment"),
	})

	return err
}
