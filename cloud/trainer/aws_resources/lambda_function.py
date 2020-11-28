import boto3
import json

# lambda re-uses execution context. 
dynamodb = boto3.resource("dynamodb")
mask_models_table = dynamodb.Table("mask_models")

def get_mask_models(event, context):
    try:
        response = mask_models_table.scan()
        
        print(f"Retrieved {len(response['Items'])} items")

        if "Items" not in response:
            return {
             "statusCode": 400,
             "message": "list is empty",   
        }
        
        return {
            "statusCode": 200,
            "message": response["Items"]
        }
        
    except Exception as e:
        print(f"Error while retrieving model list - {e}")
        return {
            "statusCode":0,
            "message": e
        }

if __name__ == "__main__":
    get_mask_models(None,None)
